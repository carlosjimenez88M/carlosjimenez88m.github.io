---
author: Carlos Daniel Jiménez
date: 2026-10-16T00:00:00-06:00
title: "When the Motif Returns, Does Its Meaning Return Too? Narrative Arcs Tested with LangGraph"
description: "The Beatles, Metallica, and Maluma featuring Farina: distinguish a semantic echo, a position that persists, and an idea contested by another voice, through close readings, embeddings, and an inspectable RAG experiment."
categories: ["Music Analysis", "Engineering"]
tags: ["embeddings", "music", "lyrics", "narrative", "RAG", "LangGraph", "evaluation"]
draft: false
images: ["/img/social/motif-returns.png"]
socialImageAlt: "When the motif returns, does its meaning return too? The Beatles, Metallica, and Maluma featuring Farina."
readerGuide:
  summary: "When an image, refrain, or position returns, what has happened to its meaning? Eight close readings across four albums distinguish constancy, changed valuation, and a reply from another voice. The RAG experiment tests whether a model actually receives the passages its interpretation requires."
  scope: "Four selected albums, 54 tracks, 53 songs with lyrics, and eight exploratory cases of recurrence. Two OpenAI embedding sizes; separate OpenAI and Anthropic readers. Textual interpretation and evidence access, rather than a ranking of artists or an analysis of audio."
  resources:
    - label: "Explore the eight returns and their evidence"
      url: "/research/motif-arcs/returns.html"
    - label: "Inspect convergence, divergence, and the sentence controls"
      url: "/research/motif-arcs/convergence.html"
---

Someone promises to change. A few passages later, the person they are addressing replies that the promise is no longer enough. When the expression of commitment returns, we recognize the words, but we now know that someone challenges them. The offer remains; its place in the conversation has changed.

That happens in “Puesto Pa’ Ti,” by Maluma featuring Farina. I wanted to read it alongside a declaration of affection that persists in The Beatles and a command that returns after a protest in Metallica. Placing these records together brings three different relationships between repetition and change into view, without assuming which should matter most.

The question is this: **when does an image, a refrain, or a position merely recur, and when does it acquire another meaning because of what happened between its appearances?**

Following an album in sequence has that appeal: something heard later can send us back to reconsider what we thought we understood. Measuring that relationship takes more care. Two passages can be close in an embedding space without expressing agreement. A fluent account can tell a story of recovery where there is only a promise. It can also gather several voices into one convenient protagonist and make the song’s conflict disappear.

I began with The Beatles: *A Hard Day’s Night*, the record containing “And I Love Her,” and *Abbey Road*. I then brought in Metallica’s *Master of Puppets* and Maluma’s *11:11*. The comparison concerns particular passages on these four records; their variety supplies different questions to ask of a return.

The engineering experiment accompanies those readings. Retrieval-augmented generation (RAG) gives a model source passages to use when answering a question. Here I check which passages actually reach the model and whether its citations refer to that supplied text. The experiment tests access to evidence under different retrieval policies; it does not isolate whether an intervening verse causes a change of meaning.

The comparison leaves a tension worth following: **the words can draw closer while the positions come apart**. A refusal can be a close neighbor of the promise it rejects. A command can return unchanged after a protest. Uncertainty about the future can coexist with attachment in the present. The figures are useful if they let us recognize these relations individually, rather than give every difference the same name.

## A repeated subject is the beginning of the question

In the earlier [study of songs about fear](/post/2026-10-11-fear-songs-embeddings/), I followed changes in what songs feared and protected. Here I want to slow down within the song. A topic such as love, autonomy, or control tells us what to look for. It leaves open what the recurring motif is *doing*: reassuring, pleading, threatening, excusing, or contesting.

That distinction has a theoretical consequence. Peter Hühn’s discussion of [events and eventfulness](https://www-archiv.fdm.uni-hamburg.de/lhn/node/39.html) separates changes in a represented world from the interpretive significance assigned to them. A wish for release does not establish an escape; recalling a loss does not mean it occurs at that point in the telling. A change can also concern what a character understands rather than what materially happens. I use these distinctions as questions to bring to the lyrics, not as an automatic scoring system.

Repetition introduces another layer. Elizabeth Margulis’s [work on musical repetition](https://collaborate.princeton.edu/en/publications/repetition/) distinguishes kinds of repetition and considers how recurring sound can be perceived differently. That helps frame the listening question, although this experiment measures lyrics only. It cannot establish how an arrangement, a singer’s delivery, or a listener’s memory transforms a refrain.

For each return, I separate the wording, its local function, the role speaking it, and the event or knowledge that might change its force. I also distinguish a narrative movement from an argument: a character’s circumstances may change, or two positions may confront one another while neither speaker gives way. Some songs sustain a lyrical situation through repetition rather than narrate a sequence of completed events.

Not every return is verbatim. A recognizable formula can be answered through a negation, as in the Maluma/Farina case. That is a transformed return, not literal repetition. Each comparison therefore needs to say what recurs before explaining what changes.

There is a further question I initially left too loose: **what, exactly, is said to change?** It might be the speaker’s commitment, the purpose of an utterance, or the reader’s interpretation after hearing another voice. A developing argument does not require every recurring sentence to acquire a new proposition.

### Four ways of coming together and moving apart

I follow both the development of positions—claims, reasons, objections, and replies—and the development of represented situations. These can intersect without moving together. An argument can become more explicit while the situation remains unresolved. Keeping both in view helps resist the temptation to supply a completed adventure where the lyric gives us an assertion under pressure.

| What we compare | What a figure can show | What still needs to be read |
| --- | --- | --- |
| Proximity between passages | A high cosine, or a curve returning toward its anchor | Passages can share a subject while one rejects the other’s proposal |
| Compatibility of positions | A stated relation between propositions with identified speakers, objects, and reference times | Not wanting something does not establish wanting its opposite |
| Development of a motif | Ordered appearances alongside the relevant reason or reply | A promise, a memory, and an accomplished event carry different commitments |
| Agreement between readers | Matching labels, or the same songs selected by both | Readers can agree on the verdict while disagreeing about who speaks |

That is why I will not draw disagreement as though it necessarily required a large geometric distance. “Something” allows present attachment and uncertainty about future growth to coexist. “Puesto Pa’ Ti” places readiness to commit against another person’s refusal. Both share an object of attention; the relations between their positions differ. Reading the scope of each claim is part of explaining that difference.

The corpus follows the specified original album editions: [13 tracks on *A Hard Day’s Night*](https://www.thebeatles.com/hard-days-night-0), [17 on *Abbey Road*](https://www.thebeatles.com/abbey-road), [eight on *Master of Puppets*](https://www.metallica.com/releases/albums/master-of-puppets-album.html), and [16 on *11:11*](https://music.apple.com/us/album/11-11/1461912465). That gives 54 tracks, 53 of them with lyrics. The model inputs are 369 ordered chunks of up to six retained lines, with at least two words per line; 363 chunk texts are unique. Single-word lines and section headings are excluded, and final chunks may be shorter. The condition that supplies all retained blocks uses this processed text; it does not provide an unabridged transcription or preserve the song’s musical sections. This filtering can omit a short repeated hook, so the experiment concerns this textual representation. “Orion” retains its place as an instrumental. The eight detailed cases below were selected after the initial readings to examine different mechanisms; their proportions do not estimate how often these mechanisms occur in an artist’s work.

## The Beatles: what remains when a declaration returns

“And I Love Her” is a useful place to begin because it resists my desire to find a dramatic development. The voice offers affection and receives tenderness; later passages assert that the attachment will endure. The declaration returns with imagery of the night sky, but the lyric does not narrate a loss, a separation, or a reconciliation between those appearances. The permanence is what the speaker claims. It is not an outcome that the song has demonstrated.

That does not make the repetition empty. Reaffirming a position is itself an action: it can reassure an addressee or sustain the speaker’s confidence. But calling that action a reversal would explain less than accepting its constancy. In the targeted experiment, both providers classify the chosen return as stable when they receive either its two endpoint blocks or all retained blocks. One condition deserves attention: the statement of permanence sits alongside the condition of remaining near the beloved. I can notice that dependence without inventing an intervening crisis.

The neighboring track “I’m Happy Just to Dance with You” offers a different case. At first, dancing is enough; the voice sets limits around physical intimacy. Near the end, a discovery of love accompanies the returning claim. The boundary does not have to disappear for the refrain to acquire another force. What was presented as sufficient activity can also become the occasion on which affection is acknowledged.

Here I have to be careful with the word *discovery*. The opening already anticipates love. The ending makes that possibility explicit rather than revealing an entirely unsuspected condition. Both providers call the selected return recontextualized when given the endpoints or all retained blocks. The textual basis is a sharpening of commitment: anticipated affection becomes an explicit acknowledgment. The passage changes the speaker’s account of the dance while preserving much of what the dance was allowed to be.

{{< figure class="wide" link="/img/motif-arcs/beatles-returns.svg" src="/img/motif-arcs/beatles-returns.svg" alt="Four Beatles close readings show stable assurance, acknowledgment of love, uncertainty about the future, and changed valuation of having no destination" caption="Interpretations by the author, with source IDs. Each row identifies the earlier target, relevant context, and the specified return. Context can lie inside a target block. The columns do not represent three successive events." >}}

On *Abbey Road*, “Something” made me reconsider a reading I wanted to defend. The [lyric](https://www.thebeatles.com/something) affirms a wish to stay in the present and later admits uncertainty about love’s future growth. I initially described that uncertainty as a qualification of the returning commitment. But the commitment was already about the present. It had not promised knowledge of the future.

That leaves a more careful reading: the song makes an epistemic limit explicit while preserving the attachment. Both all-block readers label the chosen return stable. Their vote does not settle the interpretation, but here it gives me a reason to test my own enthusiasm for finding a turn. Present commitment and uncertainty about growth can coexist. I no longer treat the passage as evidence of a demonstrated reversal or of a guarantee being withdrawn.

“You Never Give Me Your Money” offers a more direct change in evaluation. The phrase “nowhere to go” first appears amid depleted money and lost work. Shortly afterward, it is paired with a positive description of that condition. The circumstances do not need to be proven different for their valuation to change: a closed future and freedom from obligations can inhabit the same absence of a destination. The contrast is already present in two adjacent source blocks; there is no separate block between the chosen appearances. The departure scene comes afterward. It can affect a reading of the whole song, but it cannot serve as a preceding event that explains this particular return. The montage also leaves a single continuous biography unproved.

Both providers classify this target pair as recontextualized under all three evidence policies. It is the strongest Beatles case here for an explicit change in the value assigned to a recurring condition. That is more precise than saying that *Abbey Road* is simply more complex. Across the two selected records, I find constancy, acknowledgment, an explicit limit to what can be known, and changed evaluation. That does not establish a steady progression in artistic development.

## Metallica: a return can expose the persistence of power

In “Master of Puppets,” the dominating force speaks through commands and boasts. A dependent voice later challenges the promise it has received. When the controlling declaration returns, it follows that protest. A reading can hold both observations together: the master’s position has not changed, while our access to the relation has widened. We have now heard a complaint from inside it.

The distinction matters because a familiar narrative template would make the protest the beginning of emancipation. The lyric does not give me that result. The command survives its challenge. Repetition can therefore carry the experience of remaining trapped even after recognizing the deception. Recognition and release are different events.

This was also the most revealing test of my own analytical framing. In the first exploratory pass, both providers described the all-block return as recontextualized. In the revised task, both called the fixed pair of controlling passages stable, with the same retained blocks available. I changed the target definition and added binding and abstention instructions together. That comparison shows sensitivity to the revised task; it does not isolate the effect of fixing the endpoints alone.

The difference between the two readings is intelligible without choosing a winning prompt. The controller continues to assert control. An intervening voice has nevertheless made a complaint audible. What develops is the contest around the claim. Whether a listener now hears more irony, desperation, or a clearer exposure of deception is a further interpretive proposal; no listener study was run here.

{{< figure class="wide" link="/img/motif-arcs/metallica-returns.svg" src="/img/motif-arcs/metallica-returns.svg" alt="Master of Puppets places a dependent voice’s protest between returning commands; Sanitarium places a repeated plea within conflict over confinement and care" caption="The target pair is kept distinct from surrounding material. In Sanitarium, the more explicit violent imagery appears after the chosen return; it is not a preceding causal bridge." >}}

“Welcome Home (Sanitarium)” puts care and coercion in conflict. The plea to be left alone sits among confinement, intrusive judgment, institutional reassurance, and collective resistance. Its return can sound more confrontational because of that setting. Yet anger is already present early in the song, so a clean movement from passivity to agency would be too convenient. The more explicit violent imagery occurs after the selected return. It helps describe the song’s wider conflict, while leaving open what actually happens. It should not be moved backward into the space between our two targets.

Both all-block readers classify the targeted plea as recontextualized. They disagree about whether its two occurrences belong to the same local voice. Some of the generated prose is harder to defend than its label. In one reduced-context reading, an institution’s protective self-description is accepted too readily as an accurate judgment. Other readings compress the confined and collective voices. These are reasons to inspect the attributed claim, not reasons to count the shared label as a verdict.

Metallica’s two cases thus test whether the method can recognize a conflict without rewarding it with a resolution. They also expose an ambiguity in *arc*: a voice may keep its position while the song gives that position an objection, consequence, or opposing perspective. An argumentative arrangement can develop even when the commanding speaker does not.

## Maluma: the same formula enters an argument

“HP” is a necessary control against overreading change. The narrator repeatedly describes a woman who does not seek a boyfriend after betrayal. Partying, visibility to the ex, and freedom from accountability elaborate that stance. The description can contain tension—an effort to forget still organized around making the ex notice—but tension alone does not show that her stated preference reverses.

The observer’s direct offer of support comes after the specified returning refrain. I can question how the narration of her autonomy relates to the observer’s own interest. I cannot infer her acceptance or a new relationship from that offer. Both providers keep the target motif stable when given the endpoints or all retained blocks. That is an appropriate resistance to a tempting but unsupported ending.

“Puesto Pa’ Ti,” featuring Farina, gives the comparison a more explicit reply. An apologizing voice promises changed habits and commitment. The injured partner invokes abandonment, contests the adequacy of the apology, and answers the recurring formula with a negated counterpart. The negation and grammatical shift help identify an answer, rather than merely a different emotional intensity. The promise is still recognizable when it returns, but it now occupies a dispute about whether an offer entitles the speaker to reconciliation.

This is where the value at stake becomes clearer than a sentiment score would make it. One side offers readiness; the other insists on the history against which readiness must be judged. The reply is not simply a more negative version of the same feeling. It makes the right to refuse part of the argument. Nor is it one person reversing their belief: the source supplies opposed roles.

{{< figure class="wide" link="/img/motif-arcs/maluma-returns.svg" src="/img/motif-arcs/maluma-returns.svg" alt="HP distinguishes a woman’s stated autonomy from an observer’s offer; Puesto Pa Ti places a negated reply after a promise and a recollection of abandonment" caption="The return in Puesto Pa’ Ti is the clearest selected case of a recurring formula being contested by another role. In HP, the narrator’s closing offer does not demonstrate that the woman changes her preference." >}}

With all retained blocks and fixed targets, both providers identify the reply as recontextualized and attribute it to a role distinct from the initial promise. The returning block also mixes the refusal with the original pledge; no single block-level voice label can resolve every utterance in it. With only two similar retrieved blocks, the earlier exploratory pass had produced a much smoother story: both providers merged a refusal and an appeal into one speaker moving toward reconciliation. They agreed, and they erased the disagreement in the song.

That error is more consequential than missing a theme. A summary can retain the vocabulary of love and forgiveness while losing who has the power to accept or refuse. For this selected case, preserving voice changes the argument the system reports. That makes “Puesto Pa’ Ti” the clearest selected case of *dialogic* recontextualization: the promise’s meaning in the exchange depends on keeping the reply and its speaker available.

## What the comparison across albums adds

The close readings should not float free of the albums. The original two-provider annotations give a rough view of their shared subjects. The chart counts a motif once per song when its annotation cites valid source IDs. The categories were fixed in advance; they are selected salient labels, not topics discovered automatically. An unselected category is not proof that the motif is absent.

{{< figure class="wide" link="/img/motif-arcs/en/convergence-theme-overlap.svg" src="/img/motif-arcs/en/convergence-theme-overlap.svg" alt="For each album and motif, the chart separates songs selected by both readers from those selected by only one, with intersection over union" caption="A shared selection requires the same motif in the same song. Jaccard divides that intersection by the songs selected by either reader. Album denominators are 13, 17, seven songs with lyrics, and 16. An unselected motif is not demonstrated to be absent." >}}

Some broad differences survive the two readings. Control is selected in all seven songs with lyrics of *Master of Puppets* by OpenAI and in six by Anthropic. The bond motif is selected in 13 and nine tracks of *A Hard Day’s Night*, and in 14 and nine tracks of *11:11*. That makes the comparison of attachment and control plausible as an entry point. It does not establish what those subjects do in a given return.

Other differences remain unstable. Autonomy appears in three *Abbey Road* tracks for one provider and twelve for the other. In *11:11*, both readers often select control, but OpenAI distributes it mainly among means and ambivalent roles, while Anthropic usually treats it as an obstacle. A shared topic label can conceal a disagreement about agency. Averaging the two readings would make the figure neater while obscuring precisely what the close reading must resolve.

Looking at the songs themselves reveals something the totals conceal. For control on *A Hard Day’s Night*, OpenAI selects one song and Anthropic two, but **they share no selected song**. For control on *Master of Puppets*, they share six of the seven songs selected by either. For the bond motif on *Abbey Road*, totals of ten and six resolve into just four shared songs out of twelve selected by either reader. Similar-looking totals can describe readers attending to different places.

Intersection over union makes that difference visible without turning agreement into accuracy. Even a shared selection leaves another question open: of 54 song–motif pairs selected by both readers across the corpus, only 36 receive the same functional role. Calling control salient does not settle whether it is treated as a threat, a means, an obstacle, or something ambivalent. That function is part of the argument a lyric organizes.

The useful comparison is therefore between mechanisms, with a different strongest case for each question:

| Question | Most direct selected example | What the reading can support | What remains open |
| --- | --- | --- | --- |
| A recurring condition is valued differently | The Beatles, “You Never Give Me Your Money” | A lack of prospects is explicitly paired with a sense of freedom | Whether a single protagonist completes an escape |
| A repeated position meets an opposing perspective | Metallica, “Master of Puppets” | A dependent voice contests the promise before the command returns | Whether recognition brings release or the controller changes position |
| A recurring expression becomes another voice’s objection | Maluma featuring Farina, “Puesto Pa’ Ti” | A promise receives a refusal supported by reasons tied to prior conduct | Whether either speaker changes their mind or reconciliation occurs |
| The method must allow persistence | “And I Love Her” and “HP” | The specified position persists through the intervening material | What constancy contributes to the song’s effect on listeners |

This table answers which cases most directly serve the research question. It is not a quality ranking of three artists. The eight cases were chosen to make these distinctions visible, so counting their mechanisms would mostly describe my selection.

There is nevertheless a substantive comparison to make. The money song changes how a condition is valued; the master’s command persists inside a relation that can now be challenged; the duet lets another person contest the sufficiency of a promise. Revaluation, protest, and refusal are distinct developments. Compressing them into a rise in positive sentiment or a larger distance would discard what made these artists interesting to compare.

## What the embeddings show before the interpretation begins

An embedding represents a passage as a vector: a list of numbers that lets us compare it with other passages. Cosine similarity compares the directions of those vectors; a higher value means greater proximity within that model’s geometry. The score gives us a candidate relationship to investigate. It does not identify agreement, a change of speaker, or an event by itself.

The embeddings provide candidates and maps of textual proximity. For the cross-album example, I average and normalize the four unique block vectors of “And I Love Her,” then rank each Abbey Road track by its single highest-scoring block. Under that maximum-block rule, “Something” ranks first for both sizes (cosine 0.651 and 0.678). This is not an average similarity between two complete songs, and tracks with more blocks have more candidate matches. That is a useful way to find an analogy across the two records. The detailed readings distinguish an assertion of endurance from an attachment that coexists with uncertainty about growth, without requiring either to be a reversed stance.

The pair also crosses authorship: the official credits list [Lennon–McCartney for “And I Love Her”](https://www.thebeatles.com/and-i-love-her) and [George Harrison for “Something”](https://www.thebeatles.com/something). A resemblance within the Beatles’ catalog should not quietly become one songwriter’s emotional biography. The embedding has proposed a comparison between texts; explaining its significance requires keeping those distinctions available.

For the within-song question, I compare the same eight target pairs with `text-embedding-3-small` and `text-embedding-3-large`, both configured at 1,024 dimensions. Alongside each cosine similarity, the figure identifies the mechanism I propose in the close reading. That second column is an argued interpretation, rather than a discovery made by the cosine.

{{< figure class="wide" link="/img/motif-arcs/en/convergence-endpoints.svg" src="/img/motif-arcs/en/convergence-endpoints.svg" alt="Eight target-pair cosine comparisons for both encoders are placed alongside the proposed interpretive mechanisms and source IDs" caption="Cases are not ranked by narrative quality. Puesto Pa’ Ti has the highest target-pair similarity in both encoders and contains a negated reply. Absolute scores from separate encoders are not calibrated against one another." >}}

The highest of those target-pair similarities belongs to the Maluma/Farina dispute: 0.896 in the small model and 0.885 in the large. Much of the surrounding phrasing recurs, and the returning block itself mixes the refusal with the original apology. The high similarity therefore does not prove that an encoder fails to understand negation. It shows why this block-level score cannot stand in for the relation we need to establish. A very close neighbor may contain an answer that contests the first passage.

The saved results permit another concrete check. No single cutoff on cosine similarity, in either direction, separates the four target pairs both readers call stable from the four they call recontextualized under all-block exposure. In small, stable pairs range from 0.619 to 0.855 and recontextualized pairs from 0.388 to 0.896; in large, their ranges are 0.690–0.848 and 0.380–0.885. The ranges overlap. These are eight chosen cases and reader labels, rather than a general classification benchmark. Even within this selection, greater distance does not consistently identify the relation the readers label as a change.

### Following a curve without assigning it an unmeasured story

I fix the first selected appearance and compare every successive block with that anchor. Horizontal position is the block’s place in the retained text; height is its cosine similarity to the reference. A rise means renewed proximity in that geometry. A fall means less proximity. Neither movement decides whether a voice accepts, denies, or understands something.

{{< figure class="wide" link="/img/motif-arcs/en/convergence-trajectories.svg" src="/img/motif-arcs/en/convergence-trajectories.svg" alt="Eight trajectories compare successive blocks with a fixed first appearance in both encoders; rings identify the selected target pair" caption="Each song has a fixed anchor, whose similarity to itself is one. Gray regions fall outside the selected interval, preserving the location of later context. The curves show semantic echoes, not emotional arcs or agreement scores." >}}

“Puesto Pa’ Ti” shows what such a return can help us find: the reply block comes close to the initial offer, yet reading the individual turns reveals a refusal. In “Master of Puppets,” an intermediate peak also reminds us that echoes can occur outside the chosen pair. The curve does not identify controller and complainant on its own; it directs our attention to passages worth checking. In “You Never Give Me Your Money,” the selected return is farther away and the reading identifies a changed valuation within it. Two developments can have different geometries without one telling a better story.

### What similarity preserves when a claim changes

Research such as [SemAntoNeg](https://aclanthology.org/2022.blackboxnlp-1.20/) raises a useful evaluation problem: distinguishing equivalent reformulations from changes of meaning involving negation or antonyms requires more than a shared subject. It motivates a smaller diagnostic here. Rather than infer an encoder’s linguistic capacity from blocks mixing several voices, I wrote four original situations in English and Spanish. Before obtaining their vectors, I fixed a base statement, a paraphrase, a negation, and changes of participant, reference time, modality, and topic for each: 56 sentences in total.

For example, “Elena wants to stay in this city this year” is compared with its negation and with a reformulation of the same desire. The people, places, and point of observation are held fixed. Denying that desire does not assert that Elena wants to leave; changing the year or the participant does not by itself create a contradiction. These are relations between propositions, and a distance cannot decide them for us.

{{< figure class="wide" link="/img/motif-arcs/en/convergence-controlled-negation.svg" src="/img/motif-arcs/en/convergence-controlled-negation.svg" alt="Eight bilingual templates compare paraphrase-to-base and negation-to-base cosine scores for small and large, with their paired margins" caption="Margin is paraphrase cosine minus negation cosine. Small has three negative margins among eight templates, including one near tie; large has none. Four authored situations are translated into two languages. No threshold was fitted and the counts do not estimate general accuracy." >}}

In small, the negation is closer than the paraphrase in three templates: one near-tied case about staying, and both versions of permission to leave. In large, the paraphrase is closer in all eight. That is an observed advantage for large **on this criterion and these sentences**. The negation changes fewer words than the paraphrase, and the translations share situations. The diagnostic does not isolate an abstract ability to understand negation or establish a general model winner.

The modality contrast gives us another reason to keep reading after retrieval. In seven of eight templates for each encoder, replacing the assertion with a possibility produces a higher similarity than the paraphrase. Those statements can be compatible while supporting different conclusions: that someone might accept an apology does not establish that they have accepted it. This is the distinction a hurried narrative loses when it turns a wish to leave into an escape, or a promise into reconciliation.

The [convergence and divergence explorer](/research/motif-arcs/convergence.html) lets readers switch songs, follow their blocks, and compare the annotation fields. It also makes all 56 original control sentences available and their six contrasts under both encoders. A plotted point should lead back to the concrete claim that produced it.

There is a separate order problem with how we summarize vectors. Averaging a song’s block embeddings gives the same result if we reverse the sequence of those blocks. The local calculation checks this algebraic property on the saved vectors; it is not a new finding about how songs work. This is a property of mean pooling, not evidence that an embedding encoder ignores word order inside a passage. Once I have reduced a sequence to an average, however, I cannot recover its before-and-after relation from that average alone.

{{< figure class="wide" link="/img/motif-arcs/pca-variance.svg" src="/img/motif-arcs/pca-variance.svg" alt="Cumulative explained variance of the first principal components in the two embedding models" caption="The first two principal components retain 15.4% and 17.3% of variance. A reduced-dimensional picture is a partial view; its axes do not automatically represent an argument." >}}

Geometry directs us toward echoes; attribution and scope let us inspect their roles. The sentence diagnostic identifies a local difference between embedding sizes. Comparing their interpretations of songs would still require a separately defined arc task with independent reference annotations.

## Putting the return into a LangGraph evidence contract

Recent narrative RAG work gives a reason to test the sequence explicitly. [ChronoRAG](https://arxiv.org/abs/2508.18748) studies passage assembly and chronological flow in narrative question answering. [NAREOR](https://arxiv.org/abs/2104.06669) examines how changing narrative presentation affects inference. Their tasks differ from lyrical interpretation, but both help motivate a test that records access and order instead of treating a bag of relevant passages as a sufficient story.

The new experiment uses the eight specified song-level returns. Analyst-written motif descriptions guide the queries; the readers are not discovering these motifs unaided. For each, it supplies one of three contexts to each provider: two blocks selected by dense search within the song, the two manually selected endpoint blocks, or all retained song blocks in order. Dense search uses the original lyric vectors from `text-embedding-3-small`; it does not use the earlier OpenAI annotation cards. The same evidence reaches both answer providers. The input budgets differ, so this is a comparison of evidence availability, not a demonstration of better performance at equal token cost.

The LangGraph workflow retrieves candidates, applies the chosen policy, binds the question to the earlier and returning passage IDs, generates a reading, and checks its citations. Comparison between providers happens afterward. This is a bounded workflow in the sense used in the [LangGraph documentation](https://docs.langchain.com/oss/python/langgraph/workflows-agents): its steps are prescribed. Its value here is a trace we can inspect, not autonomy as a substitute for interpretation.

{{< figure class="wide" link="/img/motif-arcs/targeted-langgraph-workflow.svg" src="/img/motif-arcs/targeted-langgraph-workflow.svg" alt="The executed LangGraph workflow retrieves lyric candidates, applies an evidence policy, binds target passages, interprets and checks citation IDs before a separate comparison" caption="The implementation records exposed IDs and source checks. Checking a citation against the supplied IDs confirms that its source text was available to the model. It does not establish that the text entails the interpretation." >}}

The policies change the amount and selection of text together. All supplied blocks retain source order; there is no reordered-text arm. We therefore have **an evidence-access comparison, not an isolated test of order or of the causal contribution of a bridge**. Six-line endpoint blocks can already contain the decisive difference, as in the dancing song. Some interpretive context comes after a selected return, as in HP. All-block exposure preserves these passages by design; its coverage is not a discovery by an autonomous agent.

{{< figure class="wide" link="/img/motif-arcs/return-evidence-exposure.svg" src="/img/motif-arcs/return-evidence-exposure.svg" alt="Eight song-level exposure maps compare the retrieved top two blocks, selected endpoints and all retained song blocks" caption="Dense top-two retrieval retains both specified targets in three of eight cases. Empty gold rings identify a target whose text was not supplied. The other policies retain their targets by construction." >}}

The targeted pass contains 48 case/provider/policy cells but 42 distinct input conditions: some policies supply identical evidence to the same provider. Each completion has a 1,000-token output cap. These cells are not independent repetitions, and the experiment does not estimate run-to-run variability. The cases and prompt revisions are exploratory, following inspection of the pilot, rather than held-out confirmation.

## The model can cite what it has not read

Binding the target pair fixed one ambiguity and exposed another failure. The question named the two required passage IDs even when dense retrieval did not supply their text. In eight of sixteen dense-condition responses, the answer cited an unexposed ID. Across the ten responses missing at least one target, seven still gave a substantive relation despite an instruction to abstain. Two of the three nominal abstentions also cited unavailable material.

The observable fact is a mismatch between citation and supplied evidence. The question itself exposed the missing IDs, so a response could copy an ID without receiving its source text. The run does not tell us whether the accompanying prose came from memory, inference, or completion of a familiar pattern. It does show that instructing abstention was insufficient in this prompt. A citation check must compare against the supplied-document set, not only against IDs that exist in the corpus. The failure rate belongs to this setup; it is not a general rate for RAG.

| Evidence policy | Both targets supplied, cases | Responses citing an unexposed ID | Responses passing citation-ID and target-binding checks |
| --- | ---: | ---: | ---: |
| Dense top two | 3/8 | 8/16 | 7/16 |
| Selected endpoints | 8/8 by design | 0/16 | 16/16 |
| All retained song blocks | 8/8 by design | 0/16 | 16/16 |

The context question deserves a more precise answer than an assumption that more is always better. From selected endpoints to all retained blocks, **14 of 16 relation labels remain unchanged**. Both changes concern “Puesto Pa’ Ti”: OpenAI moves from stable to recontextualized, and Anthropic from voice contrast to recontextualized. The observed difference is concentrated in one case. In several other songs, the six-line endpoint blocks already contain the information that distinguishes their returns.

{{< figure class="wide" link="/img/motif-arcs/en/convergence-agreement.svg" src="/img/motif-arcs/en/convergence-agreement.svg" alt="Provider agreement is compared by relation, voice, and event category: seven to eight, eight to seven, and three to two matching cases" caption="Each count concerns matching provider labels among eight cases under the two policies. These are neither repeated trials nor accuracy estimates. The inherited event categories mix dimensions and do not provide a clean measure of narrative understanding." >}}

With all blocks, the readers agree on four stable and four recontextualized returns: relation agreement rises from seven to eight cases in this comparison. Voice agreement moves from eight to seven, however, and event-category agreement from three to two. A shared verdict does not guarantee a shared explanation. For “Sanitarium,” OpenAI’s distinct-voice judgment under all-block exposure is not shared by Anthropic. We still need to inspect the individual turns even when the broad label looks settled. These are observations under different evidence policies, not an isolated causal effect of adding context.

The instrument deserves criticism too. `actual` concerns realization, `recalled` temporal reference, `quoted` discourse presentation, and `hypothetical` modality. A quoted or remembered event may actually have happened. Forcing these dimensions into one choice for the case creates differences that need not be incompatible interpretations. Two matching labels out of eight help expose this inadequacy; they do not establish that a provider understands events less well. A revised annotation should separate **the proposition being evaluated, the role asserting it, its reference time, and the commitment with which it is presented**.

{{< figure class="wide" link="/img/motif-arcs/targeted-rag-comparison.svg" src="/img/motif-arcs/targeted-rag-comparison.svg" alt="Eight-by-six comparison of raw relation labels and target-voice labels across three context policies and two providers, with failed source contracts outlined in red" caption="Red frames preserve the raw answer while marking failed citation or target-binding checks. Stable, recontextualized and distinct-voice labels answer different aspects of the reading; agreement is not an accuracy score." >}}

The earlier open-target pass provides a further warning. For “Puesto Pa’ Ti,” two readers could agree about reconciliation after missing the refusal; for “Master of Puppets,” the revised target-binding task changed both all-block labels from recontextualized to stable. Several instructions changed in that revision. These observations do not supply an accuracy ranking. They make it possible to check the target, source access, and role attribution before trusting the explanation.

The [evidence explorer](/research/motif-arcs/returns.html) puts the selected passages, evidence policies, categorical readings, and failed citation-ID checks side by side. It contains original analytical paraphrases and metadata, rather than reproduced lyrics.

{{< figure class="wide" link="/img/motif-arcs/embeddings-to-rag.svg" src="/img/motif-arcs/embeddings-to-rag.svg" alt="A conceptual path connects ordered lyric passages, an embedding space, selected evidence and a checked narrative claim" caption="The three tasks remain connected: find a possible echo, recover the evidence around it, and justify the relation. An embedding neighbor supplies the first step." >}}

## A map of ideas needs more than proximity

A map of ideas needs a second layer over similarity. Each node should contain a proposition attributed to a speaker, rather than merely a nearby block of text. Each edge should name the proposed relation: reaffirms, qualifies, objects, answers, or gives a reason. Sharing an object makes the claims comparable; reading their scope explains the relation. Here the positions are drawn by hand to expose the analysis, rather than learned as model coordinates.

{{< figure class="wide" link="/img/motif-arcs/en/convergence-argument-map.svg" src="/img/motif-arcs/en/convergence-argument-map.svg" alt="Editorial proposition maps for Something, Master of Puppets, and Puesto Pa Ti distinguish persistence, compatibility, and an objection from another role" caption="Source IDs support an interpretation by the author. Edges name proposed interpretive relations and do not prove causation. Puesto Pa’ Ti separates turns that share a block. In Master of Puppets, the controlling role is inferred from how it addresses the dependent person." >}}

The map clarifies three results that a point cloud cannot decide. In “Something,” the claims have different temporal scopes and can hold together. In “Master of Puppets,” the command persists while its authority receives an objection; the conflict does not require its controlling voice to change. In “Puesto Pa’ Ti,” a reply rejects the offer. Preserving the roles avoids rewriting that exchange as one person’s emotional development. Convergence of subject and divergence of positions can occupy the same part of the analysis.

### What to investigate next, and how to test it

The next question is no longer how far a curve moved. It is **what evidence a reading needs to preserve a particular relation between ideas**. The controlled sentence comparison has already been run. The changes below require separate experiments, with criteria fixed before their outputs are inspected.

| Research question | A comparison that would test it | What we would inspect |
| --- | --- | --- |
| Does a block conceal a change of speaker? | The same utterances segmented by speaker or grouped into six-line blocks, also retaining single-word hooks | Attribution of each claim and recovery of the negated reply, using annotations for each utterance |
| Does presentation order change the interpretation? | The same passages in source order and permuted, with equal budgets and fixed targets | Changes in the justified relation and temporal citations, beyond a changed label |
| Does a passage supply the reason we attribute to it? | Identical evidence except for removing the proposed context, with a similarly sized replacement control | Whether the interpretation loses its stated support, while the target pair stays fixed |
| Does the result depend on wording or proposition? | More paraphrases, negations with different scopes, and changes of participant, time, and modality | Margins and errors by transformation, reserving new cases before rules are adjusted |
| Does the performed voice affect interpretation? | Lyrics and attributed transcript turns compared with the same evidence plus vocal or musical information | Which inferences change and which originate in the text; a listener study would provide a different kind of evidence |

LangGraph could organize that next stage: recover the appearances, check that their text is available, separate the speakers’ turns, record their propositions, and test the proposed relations against an alternative reading. If a required voice is missing, the workflow should retrieve again or stop the interpretation. If readers disagree about attribution, it should retain the alternatives and their sources. This would be a new proposition-level graph. The workflow executed here covers interpretation and citation-ID checks; it has not carried out those ablations.

A useful alternative reading would have to identify a specific difference: which claim is understood differently, and which passage supports it. More votes cannot resolve an ill-defined question. In this study, disagreement about events helped reveal a problem in the question before it could justify choosing a winning reader. A larger investigation needs to preserve that capacity to revise its instrument.

<details>
<summary><strong>Experiment record and practical limits</strong></summary>

The original album-wide pilot is preserved alongside both exploratory passes. It used `gpt-5.4-mini` and `claude-haiku-4-5-20251001`, 106 song annotations, 48 album-RAG responses, a 64-cell authored synthetic harness, and two OpenAI embedding sizes at 1,024 dimensions. The original RAG output cap of 250 tokens truncated thirteen Anthropic responses; the targeted passes use a larger, equal cap. Provider comparisons here describe these specific configurations, not the providers’ best available performance.

The original song annotations agreed least often on stance (21 of 51 pairs with valid values), and fifteen of 106 annotations failed at least one full-schema check. The motif profile above checks the source IDs for each motif separately; it does not quietly turn incomplete full annotations into literary ground truth. Withheld metadata also did not fully anonymize familiar lyrics: providers identified eleven of sixteen samples in the recognition probe.

The two targeted passes each have 48 cells and 42 distinct input conditions. They add 90 completed provider calls: in each pass, three identical concurrent requests were dispatched before the cache was populated, while three cells reused a result. Those incidental duplicates do not form a planned stochastic evaluation. The request helper now locks identical payloads to prevent that race in subsequent runs. The new diagnostic added two embedding requests for 56 original sentences, with the design fixed and reviewed before the vectors were obtained. Their estimated local cost was USD 0.0000993; the accumulated ledger is USD 0.59965462 under its configured rates, rather than a provider invoice.

Lyrics and raw provider receipts remain in the private cache. Public artifacts contain source metadata, metrics, categorical outputs and short paraphrases, with fields echoing source wording omitted. The corpus was deliberately selected, the detailed cases are selected after pilot inspection, and artist, era, language, and album form are confounded. These choices support comparison of cases, not population estimates. No model vote supplies independent reference labels for the interpretations, and no acoustic features were measured.

</details>

## What returns, and what the return allows us to understand

Across these selected cases, a motif can return with the speaker’s position intact while its role in the exchange has changed. A condition can acquire a different value; a command can face an objection; a promise can receive a refusal. The development lies in the relation between those appearances and the voices around them. Repetition alone does not tell us which relation we are hearing.

The clearest textual change in the selected Beatles cases is evaluative: a lack of direction becomes associated with relief. The dancing song makes anticipated affection explicit. “Something” is a useful restraint on the analysis: uncertainty about future growth can coexist with a commitment that was present-tense all along. A reflective reading sometimes improves by relinquishing a turn it wanted to find.

Metallica’s strongest contribution is a different distinction. A voice can recognize or contest domination while the controlling declaration persists. That does not establish liberation. The development belongs to the conflict around the claim, not necessarily to a changed belief of the speaker who repeats it. In the Maluma/Farina duet, the decisive relation is an answer: the commitment formula becomes the object of another role’s refusal. Keeping that reply changes the argument we report.

The figures support a more specific conclusion about embeddings. A semantic return can contain an opposing reply: “Puesto Pa’ Ti” has the closest target pair in both sizes. No cosine threshold separates the stable from the recontextualized returns in this selection. The authored diagnostic identifies a local difference: large ranks the paraphrase above the negation in all eight templates and small in five. Yet both also bring assertions and possibilities close together. Choosing an embedding model does not replace establishing who speaks, what time their claim concerns, and what they commit themselves to.

Additional context does not transform every interpretation. From endpoints to all retained blocks, 14 of 16 relation labels remain unchanged; both transitions belong to the Maluma/Farina duet. Overall relation agreement rises while other disagreements remain or emerge. RAG contributes by preserving and exposing relevant evidence: eight of sixteen dense-condition responses cite text that was not supplied. Studying a return requires its appearances and the reason or reply pertinent to its interpretation, followed by a check of the support for each claim. The current graph makes access to the evidence inspectable; the proposition map exposes the interpretive work the relation still requires.

These four records give us several precise ways a return can matter: as assurance, acknowledgment, revaluation, renewed domination, or a contested promise. **A shared subject, a persistent position, and agreement between people are different relations.** This exercise makes them visible separately and supplies concrete sources with which to question the reading. Recognizing the refrain is the beginning; understanding what it does when it returns requires attending to who can answer, what reasons they bring, and what remains unresolved.
