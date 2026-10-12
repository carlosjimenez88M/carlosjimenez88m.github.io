---
author: Carlos Daniel Jiménez
date: 2026-10-11T18:00:00-06:00
title: "The Evolution of Songs About Fear: An Experience Through Embeddings"
description: "Forty years of songs that talk about fear, read blind and measured with embeddings: the threat moved from the street into one's own mind, fear shifted to first-person confession, and songs learned to work through it."
categories: ["Music Analysis", "Engineering"]
tags: ["embeddings", "music", "lyrics", "narrative", "semantic-axes", "UMAP", "BERTopic", "evaluation"]
draft: false
images: ["/research/fear-lyrics/story/en/fear-locus-periods.png"]
socialImageAlt: "Composition of fear songs by period: the collective threat dominates the eighties and one's own mind grows through 2020–26"
readerGuide:
  summary: "When a song talks about fear, what is it afraid of, who feels it, how does it respond, and what does it protect? I read 120 songs from 1980 to 2026 with two independent methods: a blind paired reading and the geometry of an embedding model. Both tell the same story: fear moved from the street into the mind."
  scope: "120 songs, 12 in English and 12 in Spanish per period; 56 carry explicit or implied fear. Purposive selection, lyrics only (no audio), cutoff October 11, 2026. The percentages describe this sample, not all popular music."
  resources:
    - label: "Code and public results"
      url: "/examples/fear-lyrics-study.zip"
    - label: "Explorer: the 56 songs, their readings, and their arcs"
      url: "/research/fear-lyrics/story/en/explorer.html"
---

Far from criticizing or praising any particular era, I want to understand what songs mean, explore their relationships through embeddings, and follow how the story they tell unfolds along its narrative arc. Being able to visualize that evolution leads me to ask: **which feelings endure from one decade to the next? Which values or appreciations appear, change shape, or take on a different weight?**

The social evolution of what worries us has a change of front that begins with how we perceive the city. Fear, as in the movies, first lived outside our doors, in the street, and ended up speaking the mantra of mental fear. That is what I set out to observe at the level of the narrative arc. This time I follow songs that talk about fear, from the eighties to 2026. I want to see what remains in their worries and what changes in the way they are told: who is afraid, what they fear, what they value, and how they respond. Listening to albums, that coexistence of permanence and change is part of what draws me to following a band's musical evolution. Here I try to turn that curiosity into a reading we can test and visualize.

The short answer, in this sample of 120 songs in English and Spanish:

1. **The threat moved inward.** In the eighties, seven in ten fear songs place it in society: the street, the state, war. In the 2000s the most frequent threat is someone close, usually a partner. In 2020–26, the most frequent place is one's own mind.
2. **Fear stopped being narrated and started being confessed.** In the eighties it usually belongs to someone else ("you", "them"); since 2000, eight in ten songs voice it in the first person. In the 2010s a "we" appears.
3. **The ending changed.** Songs in which fear grows until the end decline, and songs that work through it rise. The obsessive loop, the chorus that never resolves, persists in every decade.
4. **What is protected changed.** From staying safe, to dignity, and in the last decade, to healing. Love changes least: in every period, between a third and more than half of these songs protect it.

I reached those conclusions with two measurements that never see each other: a **blind reading** of every lyric and the **geometry of an embedding model**. That they agree is what lets me state them. That they sometimes disagree is, as we will see, the most useful lesson for anyone working with embeddings.

## How to read fear in a song

The corpus has **120 songs**: 12 in English and 12 in Spanish in each period (1980–89, 1990–99, 2000–09, 2010–19, and 2020–26, the last one partial). I chose them before seeing any results: candidates associated with fear, ambiguous cases, and controls with no obvious fear motif. It is a purposive selection of hits with uneven reach, not a sample of all popular music. That is why I do not compare **how many** fear songs each decade has, which depends on who assembled the list, but **what the fear is like** within the songs that have it.

**First measurement: a blind paired reading.** Six reader processes read every lyric without artist, title, or year, with the blocks in order and repeated choruses marked. Each song received two independent readings, which record:

- whether there is fear, explicit or implied, and whether it is affirmed or denied;
- **where** the threat is: society or the state, a monster or omen, another person, the loss of a bond, time, or one's own mind;
- **who** is afraid: the voice, a "we", a "you", or a character;
- **how the voice responds**: flee, plead, confess, care, confront;
- the **state of each block** of the lyric, from threat to control, and the shape of the arc;
- **what it protects**: the value at stake.

The protocol is published. The two readings agree on whether there is fear in 95% of songs (κ = 0.90), on where the threat is in 94% (κ = 0.92), and on the shape of the arc in 82% (κ = 0.76). One important caveat: the six readers are the same assistant with the same protocol, not six people, so this agreement measures the consistency of the procedure, not independent judgment. **56 of the 120 songs** carry fear according to at least one reading; 50, according to both.

**Second measurement: the geometry.** Each block of lyrics becomes a vector with `multilingual-e5-large-instruct`, a local model, so no lyric ever leaves the machine. The model receives an instruction: *represent what the speaker fears, where the threat comes from, and how they respond*. Then I ask the space questions using anchor sentences I wrote myself. This measurement never sees the readings: it is only compared with them at the end.

## From the street to the mind

{{< figure class="wide" link="/research/fear-lyrics/story/en/fear-locus-timeline.svg" src="/research/fear-lyrics/story/en/fear-locus-timeline.svg" alt="Timeline from 1980 to 2026 with the 56 fear songs placed in six rows according to where the threat sits, from society to one's own mind" caption="Each dot is a fear song, in the year of its first release. The row shows where the two blind readings place the threat; a hollow dot means only one reading sees fear or the readings disagree on where." >}}

The figure reads top to bottom and left to right. The top row, the collective threat, clusters in the eighties and nineties. The bottom row, one's own mind, is empty until 1994 and fills up from 2000 on.

{{< figure class="wide" link="/research/fear-lyrics/story/en/fear-locus-periods.svg" src="/research/fear-lyrics/story/en/fear-locus-periods.svg" alt="Stacked bars by period: where the threat is and how the voice responds in fear songs" caption="Composition of the fear songs in each period. Equal weight for English and Spanish; a song with two different readings contributes half to each category." >}}

**The eighties: fear has an address outside.** 70% of fear songs place the threat in society, the state, or the street, and another 15% in a monster. *Somebody's Watching Me* turns surveillance into domestic paranoia. *Sunday Bloody Sunday* sings of sectarian violence in Northern Ireland. *Livin' on a Prayer*, *Fast Car*, and, beneath its mandatory cheer, *Don't Worry, Be Happy* have debts, evictions, and lost jobs in the background. In Spanish: the urban night of *Escuela de calor*, the Buenos Aires the voice flees in *En la ciudad de la furia*, and the woman declared insane in *Dr. Psiquiatra*. The voice flees, resigns itself, or cares for someone else; it confronts in fewer than a third of cases.

**The nineties: the street is still there, but the self peeks out.** Collective violence persists in *Zombie*, *Matador*, and *Gangsta's Paradise*, and a frightening loss appears (37%): *Don't Speak*, *Losing My Religion*. With *Losing My Religion* and *Hurt*, the threat inside oneself also appears for the first time in the corpus, although in both only one of the two readings sees it that way.

**The 2000s: the enemy sleeps at home.** The collective threat falls to 11%. Another person dominates (37%), almost always someone close: a partner in *Toxic*, *Time Is Running Out*, and, in Spanish, *Malo* by Bebe, released the same year Spain passed its comprehensive law against gender violence (2004); an obsessive fan in *Stan*; someone who controls and demands in *Numb*. At the same time, nu-metal and gothic rock put the monster inside the head: *Crawling*, *Bring Me to Life*, *Disturbia*. In Latin America, crises are answered with hope or prayer: *Color Esperanza*, whose album came out in November 2001, weeks before Argentina's collapse, and *A Dios Le Pido*.

**The 2010s: the street returns, in the plural.** The story is not a straight line. The collective threat rebounds to 25% with *Alright*, *This Is America*, *Latinoamérica*, and *Stressed Out*. But in the first three the voice no longer flees: it confronts, and it does so in the plural, as "we". Meanwhile, mental health moves to the center of pop with *Demons*, *Chandelier*, and *bury a friend*, which Billie Eilish described as written [from the perspective of the monster under her bed, which is herself](https://www.rollingstone.com/music/music-news/billie-eilish-bury-a-friend-video-new-lp-786599/).

**2020–26: fear as a state of mind.** One's own mind is the most frequent place (35%): *Ansiedad*, *Anti-Hero*. The body and other people's power appear in *Your Power* and *Ay Mamá*. And a passing fear grows, a moment of vertigo inside a love song: *Tengo Un Pensamiento*, *Algo tú*, *Die With a Smile*. It is the decade with the fewest cases (6.5 fear songs), so its percentages are the most fragile.

Across the full sample, the collective threat falls **12.9 percentage points per decade** (p = 0.01) and the inner threat rises **8.2 points per decade** (p = 0.02). These are weighted slopes with language effects, and the p-value comes from permuting years within each language.

### Two languages, two paths

In English, the mind appears in the nineties and accounts for a third of fear songs from 2000 on: grunge, nu-metal, and confessional pop are, among other things, the history of that move. In Spanish it does not appear until *Ansiedad*, in 2020. Before that, intimate fear runs through **the home and the partner** (*Malo*, *La Puerta Violeta*) and through **loss** (*Miedo*, *A Dios Le Pido*). It is also met more often with confrontation: between 57% and 71% of Spanish fear songs since 2000, against 40–44% in English, and the gap appears in every decade. With two to four songs per cell, this is an observation of cases, not an estimate.

## From "you" to "I", and to "we"

Whose fear is it? In the eighties, only 33% of readings attribute the fear to the singing voice; the rest belongs to a "you" being addressed (30%), as in *Thriller*, or to characters in a story (26%), as in *Cruz de Navajas* or *Hijo de la Luna*. In the 2000s, the voice confesses its own fear in 80% of cases. In the 2010s, 42% of readings find a "we": fear becomes collective again, but now it is sung by those who suffer it.

Counting "I" and "we" together, the fear belongs to the singer in 40% of songs in the eighties and in 100% in 2010–19, a rise of **12.7 points per decade** (p = 0.005). It is the strongest trend in the study, and it is not a quirk of this sample. [DeWall and colleagues](https://uknow.uky.edu/research/hit-songs-offer-window-societys-psyche) found more first-person pronouns in Billboard hits between 1980 and 2007, and [Parada-Cabaleiro and colleagues](https://www.nature.com/articles/s41598-024-55742-x), in 353,320 English lyrics, found that songs became more personal, more repetitive, and more negative. Those studies count words. Here we see what the first person does with fear: it confesses it.

## The subtopics: from war and the street to mental health

{{< figure class="wide" link="/research/fear-lyrics/story/en/fear-subtopics-context.svg" src="/research/fear-lyrics/story/en/fear-subtopics-context.svg" alt="Heatmap of eleven fear subtopics by period and bars with the embeddings' AUC for recognizing each subtopic" caption="Left: share of fear songs whose context includes each subtopic, according to the readers (a song can have several). Right: whether a question to the embedding space distinguishes the songs with that subtopic." >}}

War, police, street, poverty, and horror fill the eighties and nineties and disappear in 2020–26. Mental health is present in every decade, but it changes shape. In the eighties (22%) it is something the world does to you: the alienation of *Mad World*, the paranoia of *Somebody's Watching Me*, the psychiatric confinement of *Dr. Psiquiatra*. From 2000 on (31–44%) it is one's own mind that attacks. Love is present in every decade, between 35% and 43%: it is the most stable context of fear in popular song.

The right-hand column previews the technical part. If I ask the embedding space about a concrete **scene**, it recognizes it well: war (AUC 0.99, though with only 4 songs), control in a relationship (0.82), street and crime (0.81), monsters (0.80). If I ask about an abstract **condition**, it fails: poverty (0.50), fame (0.53), family (0.54). The model recognizes a scene better than a social structure.

## What embeddings see, and what they miss

This is the section that changed most from the first version of this study, and the one that interests me most as an engineer. In that version I tried to let the subtopics of fear emerge on their own, with BERTopic over the embeddings, and the groups turned out to be mostly language and vocabulary. Now I can explain why, and what works instead.

### 1. The first thing an embedding sees is the language

{{< figure class="wide" link="/research/fear-lyrics/story/en/fear-language-geometry.svg" src="/research/fear-lyrics/story/en/fear-language-geometry.svg" alt="Three UMAP projections of the same blocks: with E5 small and E5 large two islands appear by language; removing each language's mean mixes them" caption="Uncorrected, the five nearest neighbors of every song are in its own language. Removing each language's mean brings that share back to chance." >}}

With E5 small and E5 large, **100%** of each song's five nearest neighbors are in its own language. The map separates English from Spanish before anything else. The fear instruction alone does not fix it. Removing each language's mean, as [Libovický, Rosa, and Fraser](https://aclanthology.org/2020.findings-emnlp.150/) propose, brings that share down to 51%, that is, to chance. Any analysis of bilingual lyrics that skips this step will discover, with great confidence, that English and Spanish are different.

### 2. The free map does not answer the question

{{< figure class="wide" link="/research/fear-lyrics/story/en/fear-map-vs-question.svg" src="/research/fear-lyrics/story/en/fear-map-vs-question.svg" alt="Left: UMAP of 320 fear passages colored by where the threat sits, mixed. Right: the songs on the outside-inside axis, ordered by zone" caption="The same fear passages, without the language signal. The free map shows tendencies without borders; the axis separates the zones the readers marked." >}}

Without the language signal, free clustering of songs becomes unstable (bootstrap ARI ≈ 0.4) and barely resembles the readers' categories (NMI ≈ 0.3). The neighbors of a fear passage share a scene, vocabulary, or a chorus, but not the location of the threat. A 10-nearest-neighbor classifier gets the zone right for 59% of passages, less than the 65% you get by always answering "outside". UMAP and BERTopic remain good for exploring, but they do not measure a concept nobody asked them about.

### 3. Asking with axes

What does work is asking an explicit question. For each dimension I wrote two poles of 16 anchor sentences: 8 in English and 8 in Spanish, parallel translations, all in the first person and with explicit fear, so that only what I want to measure changes. Some examples: *I'm afraid of the stranger who keeps spying on my house* versus *I'm scared of my own thoughts*, or *I hide and wait for it to pass* versus *I'm scared, but I stand up and face it*. The axis is the difference between the two poles, an idea that comes from [Kozlowski, Taddy, and Evans's geometry of culture](https://arxiv.org/abs/1803.09288) and from [SemAxis](https://arxiv.org/abs/1806.05521). Each lyric is projected onto it.

| AUC against the readings | Outside → inside | Collective threat | Flee → confront |
| --- | ---: | ---: | ---: |
| Multilingual MiniLM | 0.77 | 0.77 | 0.54 |
| E5 small | 0.79 | 0.73 | 0.69 |
| E5 large instruct, no instruction | 0.85 | 0.80 | 0.69 |
| **E5 large instruct, with fear instruction** | **0.89** | **0.85** | **0.72** |

The large model beats the small ones, and the instruction adds 0.02 to 0.05 over the same model on every axis. On the anchors themselves, the outside→inside axis correctly classifies 100% of the sentences held out when building it.

{{< figure class="wide" link="/research/fear-lyrics/story/en/fear-compass.svg" src="/research/fear-lyrics/story/en/fear-compass.svg" alt="Scatter of the 56 fear songs on two semantic axes, colored by where the threat sits, and period means with intervals" caption="The geometry never sees the readers' labels; the colors do. Songs the readers read as inside sit on the right. On the right of the figure, period means: the outside→inside shift, with the street returning in the 2010s." >}}

Without looking at a single label, the outside→inside axis moves **+0.23 standard deviations per decade** (p = 0.01). The direction holds for all four encoders, between +0.20 and +0.27 (p ≤ 0.024), and it reproduces the readers' zigzag: forward in the 2000s, back in the 2010s, forward again. The agency axis barely moves (p = 0.21). The geometry captures **where** the threat is better than **how** the voice responds.

## The arc: how fear moves inside a song

A topic describes what a song is about; an arc describes what happens to the voice while it sings. That lyrics have arcs is already known: [Alberhasky and Durkee](https://doi.org/10.1371/journal.pone.0303188), working with 12,280 lyrics, found that songs set the scene, advance the plot, and concentrate tension in the middle, like the stories studied by [Boyd, Blackburn, and Pennebaker](https://www.science.org/doi/10.1126/sciadv.aba2196). My question is narrower: what happens to the voice's **fear** along that arc. To see it, each reader assigned a state to every block of the lyric, from threat (red) to control (blue), in the order it is sung.

{{< figure class="wide" link="/research/fear-lyrics/story/en/fear-story-strips.svg" src="/research/fear-lyrics/story/en/fear-story-strips.svg" alt="Colored strips for twelve songs from 1982 to 2022 showing the state of the voice block by block according to two readings" caption="Each row holds two strips, one per blind reading, in lyric order with repeated choruses included. Red: threat or fear; blue: defiance or agency." >}}

The strips make visible what an average hides. *Thriller* and *Somebody's Watching Me* start red and end red: the fear grows. *Crawling*, *bury a friend*, and *Anti-Hero* alternate red and pink without getting out: they are loops, the chorus as obsession. *Lose Yourself*, *Malo*, *La Puerta Violeta*, and *Ansiedad* start red and end blue: the voice works through the fear.

{{< figure class="wide" link="/research/fear-lyrics/story/en/fear-arcs.svg" src="/research/fear-lyrics/story/en/fear-arcs.svg" alt="Median trajectories of six arc shapes and their composition by period" caption="Top: how each arc shape moves along the lyric. Bottom: the share of fear songs with each shape in each period." >}}

In the eighties and nineties, a third of fear songs **escalate**. Since 2010 only 6% to 10% do (−7.7 points per decade, p = 0.06). The **turn to agency** goes from almost nonexistent (3% and 0%) to the most frequent shape in 2000–09 and 2010–19 (37% and 42%; +6.5 points per decade, p = 0.08). The **loop** barely changes: after 48% in the eighties, it hovers around 30% in each decade. And in 2020–26 **episodic** fear grows (38%; +6.9 points per decade, p = 0.01), the moment of vertigo inside a love song.

Can embeddings draw these arcs on their own? Only halfway. The correlation between the readers' state and each block's position on the agency axis is 0.32 overall, but 0.15 on average within a single song. The vector knows which zone a lyric is in, but it follows its movement poorly. The arc still needs reading.

## What the voice protects

I was interested in one more question: what makes something valuable to the one who sings. Fear is a good door, because it always fears losing something.

{{< figure class="wide" link="/research/fear-lyrics/story/en/fear-values.svg" src="/research/fear-lyrics/story/en/fear-values.svg" alt="Six bar panels with the value at stake by period: staying safe, others' lives, the love bond, the self, dignity, healing" caption="Value at stake in the readers' paraphrases, coded with explicit keyword rules. A song can protect several things." >}}

**Staying safe** (privacy, refuge, surviving the night) dominates the eighties (43%) and disappears in 2020–26. **The self** (self-control, identity, feeling alive) grows in the 2000s (22%), with *Crawling* and *Bring Me to Life*. **Dignity, freedom, and voice** reaches 71% in 2010–19: *Alright*, *This Is America*, *La Puerta Violeta*, *Take Me to Church*. **Healing and being accepted** belongs almost entirely to the last decade (32%): being able to speak and breathe in *Ansiedad*, being loved despite one's flaws in *Anti-Hero*. And **the love bond** runs through the whole period, between 35% and 57%: it is what is most protected in four of the five decades.

Read in sequence, it tells another version of the same move: first the body is protected, then identity and dignity, and finally health. This coding describes the readers' paraphrases; it is not an independent measurement, and the last decade has few cases.

## What holds and what does not

- **Holds.** With only the 50 songs in which both readers see fear, every trend holds or strengthens: inner threat +9.2 points per decade (p = 0.01), collective threat −13.5 (p = 0.008). All four encoders reproduce the outside→inside axis.
- **Weakens.** 24 of the 56 fear songs were chosen because of fear. Without them, 32 remain, and the direction holds: inner threat +3.5 points per decade, collective −10.9, axis +0.20 SD per decade. But the evidence weakens (p between 0.07 and 0.37). Part of the contrast came from the selection. What supports the **direction**, not the magnitude, is its agreement with studies of thousands of lyrics.
- **Is fragile.** 2020–26 has 6.5 fear songs, and the language splits have two to four songs per cell.
- **Is not measured.** The readers are the same model, so their agreement is not equivalent to that of different people. I read lyrics: not audio, vocal performance, or reception. A release year is not a listening year.

## What endures and what changes

**Love** endures as what fear protects, and **the loop** endures: in every decade, at least three in ten of these songs circle their fear without resolving it, because a chorus is also a way of not being able to get out.

**The place** of the threat changes: it moved from the street to the home and from the home to the mind. **The person** who fears changes, from the narrated "you" to the "I" who confesses. **The ending** changes, because songs learned to work through fear as well as suffer it. And **what is protected** changes: from surviving, to being dignified, and now to healing.

In the eighties, fear was something that happened outside and that the song told. In the 2020s, it is something that happens inside and that the song confesses, sometimes in order to get out of it.

For anyone working with embeddings, the lesson fits in two steps: **first, remove the language; then, ask.** A free map offers neighbors, and the neighbors of a lyric share a scene, vocabulary, or a chorus. An anchored direction, validated against an independent reading, answers the question you bring. And when what you want to measure is a movement inside the song, like an arc, you have to go back to reading.

<details>
<summary><strong>Method, data, and reproduction</strong></summary>

- **Corpus.** 120 lyrics with integrity hashes, from LRCLIB and Genius. The lyrics stay in a private cache; the public materials contain IDs, categories, scores, and short paraphrases without quotations. Year: first commercial release of the included recording.
- **Reading.** `story-reading-protocol.md` (in Spanish). Six reader processes, 40 cases each, randomly assigned (seed 20261012), so every song has two readings from different readers. A validator checks the schema, the IDs, and the order of the arc steps. Aggregation by fractional vote.
- **Blocks.** Up to eight lines and at least four words, in order: 1,333 blocks with repetitions, 1,137 unique.
- **Embeddings.** `intfloat/multilingual-e5-large-instruct` (1,024 dimensions, local), with and without the fear instruction. Multilingual MiniLM and E5 small for comparison. Language removed by centering (for neighborhoods and clustering) or by projecting out a direction estimated from parallel anchors (for axes and concepts).
- **Axes and concepts.** `semantic-anchors.json`, written before projecting the lyrics. Internal validation (held-out anchors) and external validation (AUC against the readings).
- **Trends.** Period means with equal weight per language. Slope per decade by weighted least squares, with language effects. p-values from 10,000 permutations of years within each language. 90% bootstrap intervals resampling songs.
- **Reproduce.** `embed_instruct.py` → `fear_story.py` → `visualize_story.py --lang en`. Public results are in `results/story-v3`.

</details>
