# --- PSYCHOMETRIC DATA INTEGRATION & NORMALIZATION ---
# Logic: Parses heterogeneous open-access scale data + codebooks into a unified
#        item-level correlation table and an item list with descriptive stats.
# Workflow: Raw CSV/TSV/SAV/RDS per scale -> per-scale intercorrelations ->
#           append to global accumulators -> export train + holdout + validation
#           parquet files.

library(tidyverse)
library(psychTools)
library(correlation)
library(stringr)
library(readxl)
library(haven)

# ---- Setup ----

dir.create("../../data/raw", showWarnings = FALSE)
options(vroom.tempdir = tempdir())
options(vroom.altrep = FALSE)

# Global accumulators (filled by intercorrelations() / append_item_list())
dat_cors <- tibble()
item_list <- tibble()
splithalf_cors <- tibble()   # filled by split_half_correlations() -- holdout/validation only

# Standardized paths used by the per-scale blocks
cb <- "/codebook.txt"
path_to_scales <- "../../data/scales/"
data_path <- "/data.csv"

# ---- Core Processing Functions ----

# Appends pairwise correlations of one scale to the global accumulator
bind_cors <- function(cor_items){
  dat_cors <<- dat_cors %>% bind_rows(cor_items)
}

# Binds the item wordings onto the global item list together with their scale
# membership. `scale_name` may be a single string (one instrument per call, the
# usual case) OR a character vector with one entry per item. The vector form is
# what the multi-scale batteries below (Bainbridge s1/s2, SurveyBot validation
# study) use, so that every item carries the instrument it actually came from
# rather than a single lumped study label.
append_item_list <- function(data, item_wordings, scale_name, scale_source) {
  stopifnot(length(scale_name) %in% c(1L, nrow(item_wordings)))
  new_rows <- item_wordings %>% mutate(scale_name = scale_name,
                                       scale_source = scale_source)
  item_list <<- bind_rows(item_list, new_rows)
}

# Main per-scale entry point: relabels columns with their item text, computes
# pairwise Pearson r (long format, upper-triangle only) and pushes both
# correlations and item-list metadata into the global accumulators.
# testing = TRUE returns the correlation tibble without mutating globals.
# `scale_name` is either length 1 or one entry per item (see append_item_list()).
intercorrelations <- function(data, dic_col, testing = FALSE, scale_name, scale_source){
  dat <- data
  colnames(dat) <- dic_col

  stopifnot(length(scale_name) %in% c(1L, length(dic_col)))

  dic_col <- dic_col %>% str_replace(., "'''", "'")

  if(!testing)
    append_item_list(data, dic_col %>% as_tibble() %>% rename(item = value), scale_name=scale_name, scale_source = scale_source)

  cor_items <- cor(dat, use = "pairwise.complete.obs") %>%
    as_tibble(rownames = "Parameter1") %>%
    pivot_longer(-Parameter1, names_to = "Parameter2", values_to = "r") %>%
    filter(Parameter1 < Parameter2)

  if(!testing)
    bind_cors(cor_items)
  else
    cor_items
}

# Splits `data` into two random respondent halves, computes item-pair
# correlations independently within each half (via intercorrelations with
# testing = TRUE, so this never touches dat_cors/item_list), and joins the
# two resulting correlation sets into one row per pair: r_half1, r_half2,
# plus each half's respondent N. This is the empirical noise-ceiling input:
# correlating r_half1 against r_half2 downstream tells you how reproducible
# the "ground truth" correlation itself is at half the sample's size, which
# (Spearman-Brown corrected) upper-bounds what any predictive model can
# achieve against the full-sample target.
split_half_correlations <- function(data, dic_col, scale_name, scale_source, seed = 42){
  set.seed(seed)
  n <- nrow(data)
  idx <- sample(seq_len(n))
  half1_idx <- idx[seq_len(floor(n / 2))]
  half2_idx <- idx[(floor(n / 2) + 1):n]

  h1 <- intercorrelations(data[half1_idx, ], dic_col, testing = TRUE,
                           scale_name = scale_name, scale_source = scale_source) %>%
    rename(r_half1 = r)
  h2 <- intercorrelations(data[half2_idx, ], dic_col, testing = TRUE,
                           scale_name = scale_name, scale_source = scale_source) %>%
    rename(r_half2 = r)

  sh <- inner_join(h1, h2, by = c("Parameter1", "Parameter2")) %>%
    mutate(n_half1 = length(half1_idx), n_half2 = length(half2_idx))

  splithalf_cors <<- bind_rows(splithalf_cors, sh)
}

# Map an arbitrary numeric column to a 1-10 scale (used before computing M/SD
# so item-level descriptives are comparable across response formats).
rescale_to_10 <- function(x, old_min, old_max) {
  ((x - old_min) / (old_max - old_min)) * (10 - 1) + 1
}

# ---- Codebook Extraction Utilities ----

# Generic extractor for Q-prefixed item lines (the OpenPsychometrics format)
extract_items_robust <- function(path) {
  lines <- readLines(path, warn = FALSE, encoding = "latin1")

  item_lines <- lines[str_detect(lines, "^Q[0-9]+[\\.\\s\\t]+")]
  item_ids <- str_extract(item_lines, "Q[0-9]+")

  # Strip IDs, repair encoding artifacts (latin1 quotes -> ASCII apostrophes)
  clean_items <- str_remove(item_lines, "^Q[0-9]+[\\.\\s\\t]+") %>% str_trim()
  clean_items <- str_replace_all(clean_items, "([a-z])\\?([a-z])", "\\1'\\2")
  clean_items <- str_replace_all(clean_items, "[^ -~]", "'")
  clean_items <- str_replace_all(clean_items, "\\s+", " ")

  names(clean_items) <- item_ids
  return(clean_items)
}

# HEXACO codebook uses trait-prefix IDs (e.g. OCrea10) rather than Q-numbers
extract_hexaco_items <- function(path) {
  lines <- readLines(path, warn = FALSE, encoding = "latin1")
  lines <- str_replace_all(lines, "[^ -~]", "'")

  pattern <- "^([A-Z][A-Za-z]+[0-9]+)\\s+(.+)$"
  item_lines <- lines[str_detect(lines, pattern)]
  matches <- str_match(item_lines, pattern)

  item_ids <- matches[,2]
  item_text <- str_trim(matches[,3])

  # Filter out metadata / validation columns mis-matched by the pattern
  is_hexaco_item <- !str_detect(item_ids, "^V[0-9]") &
    !str_detect(item_text, "ISO country code|seconds from")

  final_items <- item_text[is_hexaco_item]
  names(final_items) <- item_ids[is_hexaco_item]
  return(final_items)
}

# RIASEC codebook is tab-separated with single-letter trait + index IDs (R1...C8)
extract_riasec_items <- function(path) {
  lines <- readLines(path, warn = FALSE, encoding = "latin1")
  item_lines <- lines[str_detect(lines, "^[RIASEC][1-8]\\t")]
  parts <- str_split_fixed(item_lines, "\\t", 2)

  item_ids <- str_trim(parts[,1])
  item_text <- str_trim(parts[,2]) %>% str_remove("^[\\.\\s]+")

  names(item_text) <- item_ids
  return(item_text)
}

# ---- Multi-scale battery -> per-item scale membership ----

# The Bainbridge mega-study waves and the SurveyBot validation study are not
# single instruments but batteries of many published scales administered in one
# session. Labelling all of their items with one study-level tag would make the
# scale counts in the manuscript wrong and would prevent any scale-level
# analysis (inter-scale correlations, Cronbach's alpha, factor recovery) on
# these sets. The two helpers below recover the instrument each item belongs to.

# Bainbridge: the instrument is encoded in the variable-name prefix
# ("dtmach_1" -> "dtmach" -> SD3). Subscale prefixes of the same instrument map
# onto one name, so granularity matches the rest of the pool (one row per
# published instrument, not per facet).
scale_from_prefix <- function(col_names, lookup){
  prefixes <- str_extract(col_names, "^[^_]+")
  unmapped <- setdiff(unique(prefixes), names(lookup))
  if (length(unmapped) > 0)
    stop("Unmapped variable prefixes: ", paste(unmapped, collapse = ", "))
  unname(lookup[prefixes])
}

# SurveyBot validation study: variable-name prefixes are NOT unique there (e.g.
# "FRI_.." appears in both the Moral Foundations Questionnaire and the
# Authoritarianism Short Scale), so the instrument is taken from the SPSS
# variable label instead, which is of the form "<Scale name>: <item text>".
# Non-ASCII characters in the scale name are normalised before lookup so the
# mapping does not depend on the file's encoding.
scale_from_label <- function(labels, lookup){
  prefixes <- str_remove(labels, ":.*$") %>%
    str_replace_all("[^ -~]", "-") %>%
    str_trim()
  unmapped <- setdiff(unique(prefixes), names(lookup))
  if (length(unmapped) > 0)
    stop("Unmapped scale labels: ", paste(unmapped, collapse = " | "))
  unname(lookup[prefixes])
}

# Bainbridge wave 1 (training pool): 27 variable prefixes -> 23 instruments.
bainbridge_s1_scales <- c(
  bfi    = "BFI2",       # Big Five Inventory-2
  ip     = "IPIPNEO120", # IPIP-NEO-120
  aggr   = "BPAQ_SF",    # Buss-Perry Aggression Questionnaire, short form
  ambig  = "MSTAT_II",   # Multiple Stimulus Types Ambiguity Tolerance-II
  curi   = "CEI_II",     # Curiosity and Exploration Inventory-II
  doc    = "DCS",        # Desirability of Control Scale
  dogma  = "DOG",        # Altemeyer Dogmatism Scale
  dtmach = "SD3",        # Short Dark Triad - Machiavellianism
  dtnarc = "SD3",        # Short Dark Triad - Narcissism
  dtpsyc = "SD3",        # Short Dark Triad - Psychopathy
  empa   = "TEQ",        # Toronto Empathy Questionnaire
  flour  = "FS",         # Flourishing Scale
  gratit = "GQ6",        # Gratitude Questionnaire-6
  grit   = "GRIT",       # Grit Scale
  hope   = "AHS",        # Adult Hope Scale
  iuncr  = "IUS12",      # Intolerance of Uncertainty Scale, short form
  lone   = "DJGLS",      # De Jong Gierveld Loneliness Scale
  nfs    = "PNS",        # Personal Need for Structure
  opt    = "LOTR",       # Life Orientation Test-Revised
  satwl  = "SWLS",       # Satisfaction With Life Scale
  scpri  = "SCSR",       # Self-Consciousness Scale-Revised - private
  scpub  = "SCSR",       # Self-Consciousness Scale-Revised - public
  scsanx = "SCSR",       # Self-Consciousness Scale-Revised - social anxiety
  scont  = "BSCS",       # Brief Self-Control Scale
  seff   = "NGSE",       # New General Self-Efficacy Scale
  sest   = "RSE",        # Rosenberg Self-Esteem Scale
  vital  = "SVS"         # Subjective Vitality Scale
)

# Bainbridge wave 2 (item holdout set): 26 variable prefixes -> 21 instruments.
bainbridge_s2_scales <- c(
  bfi    = "BFI2",       # Big Five Inventory-2
  ip     = "IPIPNEO120", # IPIP-NEO-120
  aggr   = "BPAQ_SF",    # Buss-Perry Aggression Questionnaire, short form
  bas    = "BISBAS",     # BIS/BAS Scales - behavioural activation
  bis    = "BISBAS",     # BIS/BAS Scales - behavioural inhibition
  erreap = "ERQ",        # Emotion Regulation Questionnaire - reappraisal
  ersup  = "ERQ",        # Emotion Regulation Questionnaire - suppression
  empec  = "IRI",        # Interpersonal Reactivity Index - empathic concern
  empf   = "IRI",        # Interpersonal Reactivity Index - fantasy
  emppd  = "IRI",        # Interpersonal Reactivity Index - personal distress
  emppt  = "IRI",        # Interpersonal Reactivity Index - perspective taking
  grit   = "GRIT",       # Grit Scale
  hope   = "AHS",        # Adult Hope Scale
  imp    = "BIS15",      # Barratt Impulsiveness Scale, 15-item short form
  mndfn  = "MAAS",       # Mindful Attention Awareness Scale
  nfc    = "NCS18",      # Need for Cognition Scale-18
  opt    = "LOTR",       # Life Orientation Test-Revised
  pain   = "PCS",        # Pain Catastrophizing Scale
  pwb    = "SPWB",       # Ryff Scales of Psychological Well-Being
  satwl  = "SWLS",       # Satisfaction With Life Scale
  scomp  = "SCSSF",      # Self-Compassion Scale, short form
  scont  = "BSCS",       # Brief Self-Control Scale
  sest   = "RSE",        # Rosenberg Self-Esteem Scale
  smast  = "PMS",        # Pearlin Mastery Scale
  ss     = "MSPSS",      # Multidimensional Scale of Perceived Social Support
  worry  = "PSWQ"        # Penn State Worry Questionnaire
)

# SurveyBot validation study (scale-generalization set): 21 instruments, keyed
# by the scale name carried in the SPSS variable label.
surveybot_val_scales <- c(
  "Attitudes Toward AI in Defense Scale"                = "AAID",
  "Positive and Negative Affect Schedule"               = "PANAS",
  "Oldenburg Burnout Inventory"                         = "OLBI",
  "Utrecht Work Engagement Scale"                       = "UWES9",
  "Perth Alexithymia Questionnaire"                     = "PAQ",
  "Perceived Stress Scale"                              = "PSS14",
  "New Ecological Paradigm Scale"                       = "NEPS",
  "UCLA Loneliness Scale"                               = "ULS8",
  "Fear of COVID-19 Scale"                              = "FCV19S",
  "Disgust Avoidance Questionnaire"                     = "DAQ",
  "Work Gratitude Scale"                                = "WGS",
  "Center for Epidemiological Studies Depression Scale" = "CESD",
  "HEXACO-60"                                           = "HEXACO60",
  "Obsessive-Compulsive Inventory"                      = "OCIR",
  "Perseverative Thinking Questionnaire"                = "PTQ",
  "Revised Adult Attachment Scale"                      = "RAAS",
  "Authoritarianism Short Scale"                        = "KSA3",
  "Survey Attitude Scale"                               = "SAS",
  "Moral Foundations Questionnaire"                     = "MFQ",
  "Chronotype Questionnaire"                            = "CQ",
  "Big Five Inventory (BFI-10)"                         = "BFI10"
)

# Subsample large datasets to keep the correlation step tractable
shorten <- function(data){
  data %>% slice_sample(n=2000)
}

# Reset accumulators between train / holdout / validation passes
reset_accumulators <- function(){
  dat_cors <<- tibble()
  item_list <<- tibble()
  splithalf_cors <<- tibble()
}

# Write current accumulators to disk as parquet with the given prefix
export_accumulators <- function(prefix = ""){
  arrow::write_parquet(
    dat_cors %>% distinct(Parameter1, Parameter2, .keep_all = TRUE),
    paste0("../../data/raw/", prefix, "item_correlations.parquet")
  )
  arrow::write_parquet(
    item_list %>% distinct(item, .keep_all = TRUE),
    paste0("../../data/raw/", prefix, "item_list.parquet")
  )
}

# Write the split-half correlation table (if any was computed for this pass)
# to its own parquet, so downstream notebooks can estimate a noise ceiling
# without re-deriving it from raw respondent data themselves.
export_splithalf <- function(prefix = ""){
  if (nrow(splithalf_cors) == 0) return(invisible(NULL))
  arrow::write_parquet(
    splithalf_cors %>% distinct(Parameter1, Parameter2, .keep_all = TRUE),
    paste0("../../data/raw/", prefix, "item_correlations_splithalf.parquet")
  )
}

# ===========================================================================
# TRAINING SET
# ===========================================================================

# ---- Scales bundled with psychTools ----

# Eysenck Personality Inventory
intercorrelations(psychTools::epi, epi.dictionary$Content, scale_name = "EPI", scale_source = "psychTools")

# Big Five Inventory
intercorrelations(
  bfi %>% select(-c(gender, education, age)), 
  bfi.dictionary %>% filter(str_detect(ItemLabel, "gender|education|age", negate = TRUE)) %>% dplyr::pull(Item),
  scale_name = "BFI",
  scale_source = "psychTools"
  )

# Motivational State Questionnaire (drop summary scores, fix dot-separated IDs)
msq %>% select(-c(EA:exper)) %>%
  rename_with(.cols = contains("."), ~ stringr::str_replace_all(.x, "[.]", "-")) %>%
  intercorrelations(., paste0("In this moment I feel ", colnames(.)) %>% str_to_sentence(), scale_name = "MSQ", scale_source = "psychTools")

# SAPA Project Personality Inventory (skip demographics + first 10 dict rows)
spi %>%
  select(-c(age:ER)) %>%
  intercorrelations(., spi.dictionary %>% slice(11:nrow(.)) %>% pull(item), scale_name = "SPI", scale_source = "psychTools")

# Athenstaedt Gender Role Self-Concept
# intercorrelations(Athenstaedt %>% as_tibble() %>% select(starts_with("V")), psychTools::Athenstaedt.dictionary$Item[2:75], scale_name = "Athenstaedt", scale_source = "psychTools")

# ---- OpenPsychometrics scales (Q-prefixed codebook format) ----

# Humor Styles Questionnaire
humor <- read_csv(paste0(path_to_scales, "humor", data_path)) %>% select(Q1:Q32)
humor_items <- extract_items_robust(paste0(path_to_scales, "humor", cb))
intercorrelations(humor, humor_items, scale_name = "HSQ", scale_source = "OpenPsychometrics")

# Taylor Manifest Anxiety Scale
TMA <- read_csv(paste0(path_to_scales, "TMA", data_path)) %>% select(starts_with("Q"))
TMA_items <- extract_items_robust(paste0(path_to_scales, "TMA", cb))
intercorrelations(TMA, TMA_items, scale_name = "TMA", scale_source = "OpenPsychometrics")

# HEXACO-60
hexaco <- read_tsv(paste0(path_to_scales, "HEXACO", data_path)) %>%
  select(-c("V1", "V2", "country", "elapse"))
hexaco_items <- extract_hexaco_items(paste0(path_to_scales, "HEXACO", cb))
item_cols <- colnames(hexaco)[!colnames(hexaco) %in%
                                c("age", "gender", "accuracy", "country", "elapse", "V1", "V2")]
intercorrelations(hexaco %>% shorten(), hexaco_items[item_cols],
                  scale_name = "HEXACO", scale_source = "OpenPsychometrics")

# RIASEC Holland Occupational Themes
riasec <- read_tsv(paste0(path_to_scales, "riasec", data_path)) %>% select(R1:C8)
riasec_items <- extract_riasec_items(paste0(path_to_scales, "riasec", cb)) %>% 
  paste0("I would like to ", .) %>% str_to_sentence()
intercorrelations(riasec %>% shorten(), riasec_items, scale_name = "RIASEC", scale_source = "OpenPsychometrics")

# Consideration of Future Consequences Scale
CFCS_items <- extract_items_robust(paste0(path_to_scales, "CFCS", cb))
CFCS <- read_tsv(paste0(path_to_scales, "CFCS", data_path)) %>%
  select(starts_with("Q")) %>% shorten()
intercorrelations(CFCS, CFCS_items, scale_name = "CFCS", scale_source = "OpenPsychometrics")

# Depression Anxiety Stress Scales (only response columns end with "A")
DASS_items <- extract_items_robust(paste0(path_to_scales, "DASS", cb))
DASS <- read_tsv(paste0(path_to_scales, "DASS", data_path)) %>%
  select(starts_with("Q") & ends_with("A")) %>% shorten()
intercorrelations(DASS, DASS_items, scale_name = "DASS", scale_source = "OpenPsychometrics")

# Experiences in Close Relationships
ECR_items <- extract_items_robust(paste0(path_to_scales, "ECR", cb))
ECR <- read_csv(paste0(path_to_scales, "ECR", data_path)) %>%
  select(starts_with("Q")) %>% shorten()
intercorrelations(ECR, ECR_items, scale_name = "ECR", scale_source = "OpenPsychometrics")

# Empathizing / Systemizing Quotient (exclude scale totals SQ, EQ)
EQSQ_items <- extract_items_robust(paste0(path_to_scales, "EQSQ", cb))
EQSQ <- read_tsv(paste0(path_to_scales, "EQSQ", data_path)) %>%
  select(starts_with("E"), starts_with("S"), -SQ, -EQ) %>% shorten()
intercorrelations(EQSQ, EQSQ_items, scale_name = "EQSQ", scale_source = "OpenPsychometrics")

# Generic Conspiracist Beliefs Scale
GCBS_items <- extract_items_robust(paste0(path_to_scales, "GCBS", cb))
GCBS <- read_csv(paste0(path_to_scales, "GCBS", data_path)) %>%
  select(starts_with("Q")) %>% shorten()
intercorrelations(GCBS, GCBS_items, scale_name = "GCBS", scale_source = "OpenPsychometrics")

# Kentucky Inventory of Mindfulness Skills
KIMS_items <- extract_items_robust(paste0(path_to_scales, "KIMS", cb))
KIMS <- read_csv(paste0(path_to_scales, "KIMS", data_path)) %>% select(starts_with("Q"))
intercorrelations(KIMS, KIMS_items, scale_name = "KIMS", scale_source = "OpenPsychometrics")

# Machiavellianism (MACH-IV)
MACH_items <- extract_items_robust(paste0(path_to_scales, "MACH", cb))
MACH <- read_tsv(paste0(path_to_scales, "MACH", data_path)) %>%
  select(starts_with("Q") & ends_with("A")) %>% shorten()
intercorrelations(MACH, MACH_items, scale_name = "MACH", scale_source = "OpenPsychometrics")

# Multidimensional General Knowledge Test (Q*A columns store raw text answers;
# nchar / 2 converts the comma-encoded score into a numeric value)
# MGKT_items <- extract_items_robust(paste0(path_to_scales, "MGKT", cb))
# MGKT <- read_csv(paste0(path_to_scales, "MGKT", data_path)) %>%
#   select(starts_with("Q") & ends_with("A")) %>%
#   shorten() %>%
#   mutate(across(everything(), ~ nchar(.x) / 2))
# intercorrelations(MGKT, MGKT_items, scale_name = "MGKT", scale_source = "OpenPsychometrics")

# Nerdy Personality Attributes Scale
NPAS_items <- extract_items_robust(paste0(path_to_scales, "NPAS", cb))
NPAS <- read_tsv(paste0(path_to_scales, "NPAS", data_path)) %>%
  select(starts_with("Q")) %>% shorten()
intercorrelations(NPAS, NPAS_items, scale_name = "NPAS", scale_source = "OpenPsychometrics")

# Rosenberg Self-Esteem
RSE_items <- extract_items_robust(paste0(path_to_scales, "RSE", cb))
RSE <- read_tsv(paste0(path_to_scales, "RSE", data_path)) %>%
  select(starts_with("Q")) %>% shorten()
intercorrelations(RSE, RSE_items, scale_name = "RSE", scale_source = "OpenPsychometrics")

# Right-Wing Authoritarianism
RWAS_items <- extract_items_robust(paste0(path_to_scales, "RWAS", cb))
RWAS <- read_csv(paste0(path_to_scales, "RWAS", data_path)) %>%
  select(starts_with("Q")) %>% shorten()
intercorrelations(RWAS, RWAS_items, scale_name = "RWAS", scale_source = "OpenPsychometrics")

# Sexual Compulsivity Scale
SCS_items <- extract_items_robust(paste0(path_to_scales, "SCS", cb))
SCS <- read_csv(paste0(path_to_scales, "SCS", data_path)) %>%
  select(starts_with("Q")) %>% shorten()
intercorrelations(SCS, SCS_items, scale_name = "SCS", scale_source = "OpenPsychometrics")

# Adult Multidimensional Independence
AMBI_items <- extract_items_robust(paste0(path_to_scales, "AMBI", cb))
AMBI <- read_tsv(paste0(path_to_scales, "AMBI", data_path)) %>%
  select(starts_with("Q") & ends_with("A")) %>% shorten()
intercorrelations(AMBI, AMBI_items, scale_name = "AMBI", scale_source = "OpenPsychometrics")

# 16 Personality Factor Questionnaire (0 codes missing -> set to NA)
PF16_items <- extract_items_robust(paste0(path_to_scales, "16PF", cb))
PF16 <- read_tsv(paste0(path_to_scales, "16PF", data_path)) %>%
  select(A1:P9) %>%
  shorten() %>%
  mutate(across(everything(), ~ na_if(.x, 0)))
intercorrelations(PF16, PF16_items, scale_name = "16PF", scale_source = "OpenPsychometrics")

# ---- Additional OSF / OpenPsychometrics scales (mixed file formats) ----

# Self-Efficacy Fragility (own study, materials on OSF)
SEFrag <- readxl::read_xlsx(paste0(path_to_scales, "self-efficacy-security/data_final.xlsx")) %>%
  select(starts_with("item_")) %>%
  rename_with(~ str_remove(.x, "item_"))
intercorrelations(SEFrag, colnames(SEFrag), scale_name = "CSSE", scale_source = "OSF")

# Personality Inventory for DSM-5 (https://osf.io/6hwzk/overview)
PID_items <- readxl::read_xlsx(paste0(path_to_scales, "PID/codebook.xlsx")) %>%
  filter(str_detect(variable, "pid")) %>%
  dplyr::pull(label)
PID <- read_csv(paste0(path_to_scales, "PID", data_path)) %>% select(starts_with("pid"))
intercorrelations(PID, PID_items, scale_name = "PID", scale_source = "OSF")

# Symptom Checklist 90 Revised (https://osf.io/q5rgb/overview)
SCL90R_items <- extract_items_robust(paste0(path_to_scales, "SCL90R", cb)) %>% 
  paste0("During the past 7 days, including today, I was bothered by ", .) %>% str_to_sentence()
SCL90R <- read_xlsx(paste0(path_to_scales, "SCL90R/data.xlsx")) %>%
  mutate(across(everything(), as.numeric))
intercorrelations(SCL90R, SCL90R_items, scale_name = "SCL90R", scale_source = "OSF")

# Psychological Strain Scales (https://osf.io/q5rgb/overview)
PSS_items <- extract_items_robust(paste0(path_to_scales, "PSS", cb))
PSS <- read_csv(paste0(path_to_scales, "PSS/data.csv")) %>%
  mutate(across(everything(), as.numeric))
intercorrelations(PSS, PSS_items, scale_name = "PSS", scale_source = "OSF")

# Comprehensive Autistic Inventory + Adult ADHD Self-Report (https://osf.io/qtngb/overview)
CATI_ASRS_items <- extract_items_robust(paste0(path_to_scales, "CATI_ASRS", cb))
CATI_ASRS <- read_csv(paste0(path_to_scales, "CATI_ASRS/data.csv"), skip = 1) %>%
  select(contains("ASRS"), contains("CATI"), -`Education level`)
intercorrelations(CATI_ASRS, CATI_ASRS_items, scale_name = "CATI_ASRS", scale_source = "OSF")

# General Attitudes towards Artificial Intelligence Scale (SPSS .sav with value/
# format labels; zap_* strips SPSS metadata that would break downstream numerics)
GAAIS <- haven::read_sav(paste0(path_to_scales, "GAAIS/data.sav")) %>%
  select(Pos1:Trust, -starts_with("BLANK")) %>%
  select(-c(21:57, 78:87))
colnames(GAAIS) <- sapply(GAAIS, function(x)
  attr(x, "label") %||% colnames(GAAIS)[which(sapply(GAAIS, identical, x))])
GAAIS <- GAAIS %>% zap_labels() %>% zap_label() %>% zap_formats() %>% zap_widths() %>% 
  select(-CorporateDistrust1)
intercorrelations(GAAIS, colnames(GAAIS), scale_name = "GAAIS", scale_source = "Supplementary Data, Journal")

# Chronic Pain - Emotional / Trauma Survey
C_PETS_items <- extract_items_robust(paste0(path_to_scales, "C-PETS", cb))
C_PETS <- read_csv(paste0(path_to_scales, "C-PETS", data_path), skip = 2) %>% select(-c(1:4))
intercorrelations(C_PETS, C_PETS_items, scale_name = "CPETS", scale_source = "OSF")

# Educational Physical Technology Engagement and Performance Scale
EPTEPS_items <- extract_items_robust(paste0(path_to_scales, "EPTEPS", cb))
EPTEPS <- read_csv(paste0(path_to_scales, "EPTEPS", data_path)) %>% select(UE1:MO5)
intercorrelations(EPTEPS, EPTEPS_items, scale_name = "EPTEPS", scale_source = "OSF")

# Vanity Scale (reverse-code items ending in "R" on a 1-5 Likert)
Vanity_Scale_items <- extract_items_robust(paste0(path_to_scales, "Vanity_Scale", cb))
Vanity_Scale <- read_sav(paste0(path_to_scales, "Vanity_Scale/data.sav")) %>%
  select(I1:I22) %>%
  zap_formats() %>%
  mutate(across(ends_with("R"), ~ 6 - .x))
intercorrelations(Vanity_Scale, Vanity_Scale_items, scale_name = "VanityScale", scale_source = "OSF")

# Multidimensional Sexual Self-Concept Questionnaire
# (OpenPsychometrics raw-data repository, not OSF -- fixed to match Table A1)
MSSCQ_items <- extract_items_robust(paste0(path_to_scales, "MSSCQ", cb))
MSSCQ <- read_tsv(paste0(path_to_scales, "MSSCQ", data_path)) %>% select(starts_with("Q"))
intercorrelations(MSSCQ, MSSCQ_items, scale_name = "MSSCQ", scale_source = "OpenPsychometrics")

# Hypersensitive Narcissism Scale + Dirty Dozen
# (OpenPsychometrics raw-data repository, not OSF -- fixed to match Table A1)
HSNS_DD_items <- extract_items_robust(paste0(path_to_scales, "HSNS+DD", cb))
HSNS_DD <- read_tsv(paste0(path_to_scales, "HSNS+DD", data_path)) %>%
  select(starts_with("H"), starts_with("D"))
intercorrelations(HSNS_DD, HSNS_DD_items, scale_name = "HSNS_DD", scale_source = "OpenPsychometrics")

# Short Dark Triad
# (OpenPsychometrics raw-data repository, not OSF -- fixed to match Table A1)
SD3_items <- extract_items_robust(paste0(path_to_scales, "SD3", cb))
SD3 <- read_tsv(paste0(path_to_scales, "SD3", data_path)) %>% select(-country, -source)
intercorrelations(SD3, SD3_items, scale_name = "SD3", scale_source = "OpenPsychometrics")

# Bainbridge mega-study s1 (item texts via the authors' RDS label dict, with
# "I am someone who - X" / "I - X" prefixes flattened into proper sentences).
# This wave is a battery of 23 published instruments, so scale membership is
# recovered per item from the variable-name prefixes rather than tagging the
# whole wave with one study-level label.
labels_bb <- read_rds(paste0(path_to_scales, "Bainbridge/label.rds"))
bainbridge <- read_csv(paste0(path_to_scales, "Bainbridge", data_path)) %>%
  select(-c(1, ac_1:Consent_T_Click.Count))
bainbridge_items <- unname(labels_bb$s1[colnames(bainbridge)]) %>%
  str_replace(" - ", " ") %>%
  str_to_sentence()
bainbridge_scales <- scale_from_prefix(colnames(bainbridge), bainbridge_s1_scales)
intercorrelations(bainbridge, bainbridge_items, scale_name = bainbridge_scales, scale_source = "OSF")

# SAPA 696-item public release
sapa_items <- read_csv(paste0(path_to_scales, "SAPA/iteminfo696.csv"),
                       locale = locale(encoding = "latin1")) %>%
  select(Item) %>% 
  mutate(
    Item = str_remove_all(Item, "\\[|\\]")
  ) %>% 
  dplyr::pull() %>% paste0("I ", .) %>% str_to_sentence()
sapa <- read_csv(paste0(path_to_scales, "SAPA/data.csv")) %>% select(-c(RID:p2occIncomeEst))
intercorrelations(sapa, sapa_items, scale_name = "SAPA", scale_source = "SAPA")

# ---- Export training set ----

item_list <- item_list %>% distinct(item, .keep_all = TRUE)

dat_cors <- dat_cors %>% distinct(Parameter1, Parameter2, .keep_all = TRUE)

export_accumulators(prefix = "")

# ===========================================================================
# HOLDOUT SET (Bainbridge s2 - cross-scale, item-disjoint from training)
# ===========================================================================

reset_accumulators()

bainbridge_holdout <- read_csv(paste0(path_to_scales, "bainbridge_2", data_path)) %>%
  select(-c(1, ac_1:DB_T_Click.Count))

stopifnot(all(colnames(bainbridge_holdout) %in% names(labels_bb$s2)))
bainbridge_holdout_items <- unname(labels_bb$s2[colnames(bainbridge_holdout)]) %>%
  str_replace(" - ", " ") %>%
  str_to_sentence()
# Wave 2 is a battery of 21 published instruments (see bainbridge_s2_scales).
bainbridge_holdout_scales <- scale_from_prefix(colnames(bainbridge_holdout), bainbridge_s2_scales)
intercorrelations(bainbridge_holdout, bainbridge_holdout_items,
                  scale_name = bainbridge_holdout_scales, scale_source = "OSF")

# Empirical noise-ceiling input: independent item-pair correlations from two
# random respondent halves of the same sample (see split_half_correlations()).
split_half_correlations(bainbridge_holdout, bainbridge_holdout_items,
                        scale_name = bainbridge_holdout_scales, scale_source = "OSF")

export_accumulators(prefix = "holdout_")
export_splithalf(prefix = "holdout_")

# ===========================================================================
# VALIDATION SET (SurveyBot validation study - completely separate scales)
# ===========================================================================

reset_accumulators()

# Item wordings live in SPSS variable labels; pattern strips the "TraitX: " prefix
sb_val <- read_rds(paste0(path_to_scales, "surveybot_val_study/data.rds")) %>%
  as_tibble() %>%
  select(AAID_01:CQ_16)

sb_labels <- map_chr(sb_val, ~ attr(.x, "label") %||% NA_character_)

# keep real items only: has a label, and not an attention check
keep <- !is.na(sb_labels) &
  !str_detect(sb_labels, "To ensure data quality, please choose")

sb_val       <- sb_val[keep]                          # same columns dropped
sb_val_items <- str_remove(sb_labels[keep], "^.*: ")  # strip prefix on survivors

# The stripped prefix is the instrument name; recover it as the per-item scale
# label so the 21 validation instruments are not collapsed into one study tag.
sb_val_scales <- scale_from_label(sb_labels[keep], surveybot_val_scales)

sb_val <- sb_val %>% zap_formats() %>% zap_label() %>% zap_labels()
intercorrelations(sb_val, sb_val_items, scale_name = sb_val_scales, scale_source = "OSF")

# Empirical noise-ceiling input: independent item-pair correlations from two
# random respondent halves of the same sample (see split_half_correlations()).
split_half_correlations(sb_val, sb_val_items, scale_name = sb_val_scales, scale_source = "OSF")

export_accumulators(prefix = "validation_")
export_splithalf(prefix = "validation_")
