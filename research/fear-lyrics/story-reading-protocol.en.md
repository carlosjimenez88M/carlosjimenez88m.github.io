# Reading fear along the narrative arc: protocol v3

English translation of `story-reading-protocol.md`, which is the version the readers received. Date: October 12, 2026. Corpus cutoff: October 11, 2026. The same procedure applies to all 120 songs.

For the historical question, this reading replaces the annual review that was left incomplete (104 of 240 readings). It answers three questions per song: **where the threat is**, **how the voice responds**, and **how the lyric moves between those states**. Earlier readings (the 40-song panel and the partial annual review) are kept unchanged.

## Design

- Six readers (L1–L6): separate processes of the same assistant with the same protocol. Each reader gets 40 cases, and each song gets exactly **two readings** from different readers. Random assignment with seed 20261012. They are not human annotators or independent providers.
- Cases are anonymized: no artist, title, year, declared language, selection label, or embedding results. Anonymizing is not blinding: in a later test, the same model (Claude Opus 5.5, Claude Code subagents with default sampling) identified 110 of the 120 lyrics from these same batches (`results/story-v3/recognition_summary.csv`). Blocks are delivered in order (`b01`, `b02`, …); a repeated block appears as `repeat_of`, without text.
- Only the text is read. If a reader recognizes the song, it does not use biography, music video, or reception.
- Public outputs contain categories, IDs, and short paraphrases in Spanish; never quotations or reconstructed lines.

## Fields

| Field | Values |
| --- | --- |
| `fear_reference` | `explicit` (fear named, even if denied), `implied` (a threatening or fearful situation supported by the text without naming fear), `absent`, `uncertain`. Sadness, heartbreak, or desire are not enough. |
| `fear_polarity` | `affirmed`, `denied`, `hypothetical`, `mixed`, `absent`, `unclear` |
| `who_fears` | `speaker`, `speaker_collective`, `addressee`, `third_party`, `nobody`, `unclear` |
| `threat_locus` | Where the lyric places the source of the threat: `external_person` (a specific someone: stalker, partner, pursuer), `external_collective` (society, state, war, city, economy, system), `supernatural_fiction` (monsters, narrated horror, fantasy), `relational_loss` (losing someone or a bond), `internal_self` (one's own mind, body, or self as the threat), `existential_time` (future, death, passing time, uncertainty without an agent), `none`, `uncertain` |
| `threat_locus_secondary` | Same set, or `null` |
| `response` | The voice's dominant response: `flight_avoidance`, `paralysis`, `plea_prayer`, `defiance_confrontation`, `confession_naming`, `care_protection`, `resignation`, `transformation_agency`, `none`, `unclear` |
| `block_states` | One state per block, repetitions included: `threat`, `fear`, `avoidance`, `plea`, `defiance`, `agency`, `care`, `confession`, `loss`, `desire`, `celebration`, `reflection`, `filler` |
| `arc_shape` | `loop` (the fear situation repeats without change), `escalation` (it intensifies toward the end), `release` (it eases through outside relief, comfort, or time), `turn_to_agency` (the voice moves from fear to confronting or deciding on its own), `collapse` (from hope or control toward fear, loss, or resignation), `episodic` (fear appears in a single place, without development), `no_fear_arc` |
| `arc_steps` | 2–4 steps `{"ids":[...],"reading":"≤12 words"}` with strictly increasing IDs between steps; empty if `no_fear_arc` |
| `threat_object` | ≤8 words: what is feared |
| `value_at_stake` | ≤8 words: what the voice values, protects, or wants to transform |
| `context_markers` | Up to 3: `war_conflict`, `state_police_surveillance`, `urban_violence_crime`, `economy_class`, `migration_borders`, `gender_violence`, `mental_health`, `media_technology_fame`, `religion_faith`, `nature_disaster`, `romance_sex`, `family_home`, `horror_fiction`, `none` |
| `confidence` | `high`, `medium`, `low` |
| `counterreading` | ≤20 words: the strongest alternative reading |

Rules: `threat_locus`, `response`, and `arc_shape` are recorded even when fear is `uncertain`; with `absent`, use `none` / `no_fear_arc` unless there is a threat without fear. `internal_self` requires the text to present one's own mind, behavior, or identity as a danger (a first-person voice is not enough). `external_collective` requires a social agent or condition, not an individual.

## Aggregation

Every reading is kept. A category counts as **agreed** when both readings match; disagreements are reported and, for the figures, the song contributes a weight of 0.5 to each proposed category (fractional vote). Time series are shown with song counts, split by language, and with bootstrap intervals over songs. Inter-reader agreement (Cohen's κ) is published per field.

`block_states` feeds the arc: each state is mapped to an ordinal threat ↔ control scale and interpolated to ten positions of lyric time. Embeddings are compared with these readings; they are not used to create them.

The readers' paraphrases are in Spanish. The English explorer shows translations of them (`results/story-v3/paraphrases_en.json`).
