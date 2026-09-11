# Execution notes

- Collection completed with 56/56 exact normalized artist/title matches. No
  identical lines were shared between the separately collected Thrown Away and
  Tightrope inputs; hidden-track text was not duplicated across those units.
- The first question-generation request was rejected by the provider because
  JSON mode requires the word JSON in the messages. The prompt was corrected;
  previously completed annotation and summary calls were reused.
- Some proposed question pairs violated the predeclared cross-half/distance
  constraints. A bounded three-attempt validator requested corrected sets before
  answer evaluation. These extra preparation calls remain in usage totals.
- The final questions reveal coarse positional constraints (first/second half or
  separation), although the initial prompt requested no positions. They reveal
  no exact target IDs or track titles. Treat this as a protocol deviation and a
  task framing limitation, not as successful enforcement of every natural
  language instruction. The same frozen questions are used for every policy.
- The memory policy comparison uses a fixed retrieval heuristic rather than
  selecting its parameters on observed answer scores. Follow-up policies must
  be identified separately from this pilot.

- After inspecting case beatles-recent3-change-0, added a post-hoc diagnostic
  of whether answer citations span both album halves and a 90-word compliance
  count. These are separate exports; original judge metrics are unchanged.
  Neither diagnostic validates the semantic claim-to-source relationship.
