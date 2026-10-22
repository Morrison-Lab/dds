# Authoring lecture content

How to write and edit the lecture pages under `chapters/`,
for people and agents alike.
The general rules --- divs and their titles, slide breaks, subfiles, citations, math,
code chunks, figures and tables --- are in
[`.github/copilot-instructions.md`](.github/copilot-instructions.md),
which follows [`Morrison-Lab/rme`](https://github.com/Morrison-Lab/rme)'s instructions.
This file adds what is specific to this course.
It was copied from
[`Morrison-Lab/lds`](https://github.com/Morrison-Lab/lds)'s `AUTHORING.md` (DATA 571),
whose directions carry over unless a rule below says otherwise:
Ezra's directions for DATA 571,
and the conventions this course took from
[`Morrison-Lab/epi204`](https://github.com/Morrison-Lab/epi204).
`CLAUDE.md` covers the repository itself.

The general rules for writing teaching materials are in psw's
[Writing for teaching](https://morrison-lab.github.io/psw/chapters/teaching.html) chapter,
and this file does not repeat them:
exercise-first lectures, one term per `#def-` div, no nested theorem-type divs,
notation and synonyms, real data and computed numbers,
[structured data files](https://morrison-lab.github.io/psw/chapters/teaching.html#store-structured-data-in-structured-formats),
accessible figures, slide structure, and crediting other courses.
Where a section below restates one of them, it adds this course's own detail
and links to the psw rule.
The rme and epi204 rules were read from those repositories' `CLAUDE.md` and
`.github/copilot-instructions.md` on 2026-09-24;
the exercise-first rules are Ezra's, from 2026-09-25,
and were not written down anywhere before this file.

## Lectures are built from exercise and solution pairs

The general rule is psw's
[build the lecture from exercises and solutions](https://morrison-lab.github.io/psw/chapters/teaching.html#build-the-lecture-from-exercises-and-solutions).
Ezra, 2026-09-25:
fit as much of each lecture as possible into exercise and solution pairs,
to keep students working rather than watching.

- **Every data manipulation or analysis is an exercise.**
  Pose the task in an `#exr-` div,
  give students time to try it,
  then show and discuss good solutions in the matching solution div.
  Do not show the code first and explain it afterwards.
- **Every calculation, derivation or proof starts as an exercise.**
  Before showing any math,
  ask students to work it out in an `#exr-` div;
  the worked steps go in the solution.
- **Ask for every definition and theorem before stating it.**
  Wherever possible, precede a `#def-` or `#thm-` div with an exercise
  that asks the question it answers.
  The solution gives an informal answer in plain words,
  and the formal div follows after the solution div, never inside it.
  Lead up to it with earlier exercises when that helps.
  For example, one exercise asks "What is the goal of machine learning?",
  and its solution answers that the goal is almost always
  accurate generalization to, or prediction of, new data.
  The next asks "What is generalization?",
  its solution answers informally,
  and the `#def-generalization` div follows that solution.
  In the same way, the solution to "What is machine learning?" can say
  that all the algorithms shown learn from data,
  so machine learning might be the field of algorithms that learn from data;
  the `#def-machine-learning` div then follows that solution.
- **Derive a theorem in a solution, then state and prove it separately.**
  The exercise asks for the result, and its solution derives it.
  A separate `#thm-` div after that solution states the result,
  and a `proof` div right after the theorem links back to the solution,
  as rme's `_subfiles/logistic-regression/_sec_odds_fn.qmd` does:

  ````markdown
  ::: {#exr-prob-to-odds}
  Find a general formula for converting probabilities into odds.
  :::

  {{< slidebreak >}}

  ::: {#sol-prob-to-odds}
  (the derivation)
  :::

  {{< slidebreak >}}

  ::: {#thm-prob-to-odds}
  #### Odds as a function of probability
  (the statement)
  :::

  ::: proof
  By @sol-prob-to-odds.
  :::
  ````

- **Break a complex task into subtasks.**
  Each bottom-level prompt gets its own `#exr-` div and its own solution,
  in order,
  rather than one exercise with parts (a) through (e)
  (epi204: "Use `#exr` divs for each bottom-level prompt instead of Markdown headers").
  Later subtasks may use earlier results;
  cross-reference them with `@exr-...`.
- **Put a slide break between an exercise and its solution,**
  so the prompt is on screen alone while students work.
- Give the solution the exercise's slug with `sol-` in place of `exr-`.
- Put a hint in a `::: {.callout-tip}` div titled `#### Hint`, inside the exercise div,
  not as `*Hint:*` text (epi204).

### Solutions here must not answer graded work

An in-class exercise whose solution is discussed in lecture is not graded,
so its solution belongs in these notes.
But this repository is what students read:

- An exercise that also appears in a homework, quiz, exam or handout
  never has its solution here, in any wording or with any numbers.
  Check `dlg`'s graded material before writing one;
  pick a different example, or leave the solution in `dlg`.
- Do not copy a question from `dlg`'s question bank (`exercises/` there) into a lecture.
- A worked general method that reduces a graded question to substitution counts as its answer.
  In lds, the neural networks chapter's logic gates are the example:
  the notes may say which gates one unit can compute, and why,
  but not give the weights or the recipe that a graded handout asks for.

## Data and code

- **Give code examples in Python, with PyTorch**, as executed `{python}` chunks.
  This differs from lds, which pairs R and Python and adds Julia:
  deep learning is taught and practised in PyTorch, and the homework uses it.
  Where a lecture goes under the hood and implements a method,
  implement it first in NumPy, then show the PyTorch building block that replaces it.
- **Keep every executed chunk small enough to run in seconds on a CPU**,
  since CI renders every page on a runner without a GPU:
  tiny networks, subsets of MNIST, short training runs with a fixed seed.
  Larger runs belong in the homework, not in the rendered notes.
- **Never type or paste output** (Ezra, 2026-09-28).
  What the page shows under a chunk is what the chunk printed at render time,
  and a number in the prose comes from inline code.
- **Give students the real data wherever a dataset is named** (2026-09-25).
  Keep a copy in `chapters/data/`, list it under the page's `resources:` so students can download it,
  and put the loading and looking in exercises,
  as `chapters/course-overview.qmd` does for Spambase.
  The prose names the original source, links where it is published, and cites it.
- Real data and computed numbers follow psw's
  [use real data and compute every number](https://morrison-lab.github.io/psw/chapters/teaching.html#use-real-data-and-compute-every-number).
- Show a dataset before using it:
  print a look at it (`str()` or `dplyr::glimpse()` in R, `.head()` in Python) in a chunk,
  as rme's logistic-regression chapter prints the `wcgs` data.
- Figures follow psw's
  [make figures accessible](https://morrison-lab.github.io/psw/chapters/teaching.html#make-figures-accessible).
- An interactive figure (an `{ojs}` widget and its inputs) is a figure too (2026-09-26):
  wrap it in a `::: {#fig-...}` div with a caption,
  and give the plot alt text (Observable Plot's `ariaLabel`).

## Notation

- **Say which notation we use, and which others a reader will meet** (2026-09-25;
  psw: [state the notation and its alternatives](https://morrison-lab.github.io/psw/chapters/teaching.html#state-the-notation-and-its-alternatives)).
  When introducing a symbol or convention,
  state the notes' choice and name the common alternatives in other sources,
  in a definition div or a callout beside it.
  The math prerequisites' natural numbers are the example:
  some sources include 0 and some exclude it,
  and $\mathbb{N}_0$ and $\mathbb{N}^+$ make the choice explicit.
- Name a term's synonyms and near-synonyms, such as learning, training and fitting,
  in its definition div or a callout beside it, and say which one the notes use.
- **Reconcile alternative representations for the same input** (2026-09-30).
  When introducing multi-dimensional inputs such as images,
  explicitly reconcile the representational conventions across model families:
  classical models (like linear models) flatten a $16 \times 16$ image
  into a 1-D feature vector $x \in \mathbb{R}^D$ ($D=256$),
  whereas spatial architectures (like convolutional networks)
  preserve the native grid structure as a multi-dimensional array
  ($D_1 \times D_2$ matrix or $H \times W \times C$ tensor).
  Never present these representations adjacent to each other
  for the same running example without explicitly bridging them.
- **Predictor and design matrix dimensions** (2026-10-07).
  Use $p$ for the number of raw predictors ($\vx \in \mathbb{R}^p$).
  Use $p^*$ for the number of elements in the design vector $\phi(\vx) \in \mathbb{R}^{p^*}$
  and the number of columns in the design matrix $\design(\mX) \in \mathbb{R}^{n \times p^*}$.
  For an affine model with an intercept, $p^* = p + 1$.
  Parameters match the design space: $\vbeta \in \mathbb{R}^{p^*}$.
  Hat matrix projection is $H = \design(\design^\top \design)^{-1} \design^\top \in \mathbb{R}^{n \times n}$,
  with individual leverage $h_i = H_{ii} = \phi(\vx_i)^\top (\design^\top \design)^{-1} \phi(\vx_i)$
  and average leverage $\bar{h} = p^*/n$.
  Polynomial feature expansions of degree $d$ produce $p^* = \binom{p+d}{d}$ terms.

## Standing directions from Ezra

Each applies to every page, not only the one that prompted it.

- **Name a term's synonyms where the term is defined** (2026-09-25).
  When a piece of jargon has synonyms or near-synonyms,
  say so in its `#def-` div, or in a callout beside it.
  Learning, training and fitting are the example Ezra gave;
  the notes also carry weights and parameters and coefficients,
  inner product and dot product,
  features and predictors and covariates,
  and loss and cost and objective.
  A student who meets one spelling here and another in a textbook
  has no way to tell whether the difference is a distinction,
  and will look for one that is not there.
  Near-synonyms need the seam named rather than flattened:
  stationary and critical describe the same point
  only where the function is differentiable.
  Two senses of one word earn a callout for the opposite reason,
  as the bias term $b$ and statistical bias do.
- **Organize the notes by topic, not by week** (2026-09-18).
  A chapter covers one subject, and the calendar says which week teaches it.
- **Only the calendar discusses the schedule** (2026-10-06).
  No page but `calendar.qmd` says when something is taught:
  no week or day in a subtitle, no "on Tuesday", "Wednesday's lecture",
  "next week" or "the last week of the course".
  Point to the page that covers a topic instead,
  as in "[Gradient Descent](gradient-descent.qmd) starts on that."
  Each reshuffle of the schedule left such references stale (#418, #424).
  Citing a source's own week numbering ("the Week 6 lecture notes") is not a schedule reference.
  Reading order is fine ("the next chapter", "so far"),
  but say "these notes", not "the course", when the order is the point:
  "the rest of these notes", not "the rest of the course".
- **Flesh the notes out beyond earlier offerings** (2026-09-18, 2026-09-24).
  Start from past years' material,
  then add from the collected sources:
  Brian Hutchinson's CSCI 481/581 lectures, labs and project (used with his permission),
  the book PDFs in [`Morrison-Lab/mlr`](https://github.com/Morrison-Lab/mlr),
  and *Deep Learning* [@goodfellow2016deep].
  His video lectures sit behind a login that his site asks not be shared,
  so never link them from these notes.
- **Include interactive content** (2026-09-24).
  [`Morrison-Lab/ISLP_labs`](https://github.com/Morrison-Lab/ISLP_labs)
  and [`Morrison-Lab/ISLP`](https://github.com/Morrison-Lab/ISLP)
  are the first place to look for ideas.
- **The course is CSCI 481**, and Spring 2027 has no CSCI 581 section (2026-10-22).
  Never write "481/581".
- **Use Quarto's own mechanism, set up the way rme does it, before writing a new one** (2026-09-24).
  The last-modified date is the example: `date: last-modified` at the project level.
- **PDF handouts are shared on their own through Canvas**, so:
  no table of contents (2026-09-24, [#76](https://github.com/Morrison-Lab/lds/issues/76));
  no links to other pages, which `filters/unlink-pages.lua` removes (2026-09-23);
  a link to a data file on the site becomes its full site URL, since students cannot open the repository;
  every page shows its last-updated date, as the web page and slides do
  (2026-09-24, [#86](https://github.com/Morrison-Lab/lds/issues/86));
  and dense tables stay readable, with rules or space between rows (2026-09-24).
- **Keep every mention of the start-of-quarter survey current** (2026-09-24):
  when the link or the questions change, update every page that mentions it,
  and check it against the live form.
- **List comparable courses and their materials in the syllabus** (2026-09-24).
- **Decompose heavily, and when it is a judgement call, decompose further**
  (2026-09-25).
  The chapter `.qmd` is a spine of headings, slide breaks and `{{< include >}}` calls,
  and each unit is its own subfile
  named `_<stem>_<type>.qmd` with the div type last
  (`_def`, `_exr`, `_sol`, `_thm`, `_exm`, `_fig`, `_rem`),
  so the files of one idea sort together:
  `_backfitting-update_def.qmd`, `_backfitting-update_exr.qmd`, `_backfitting-update_sol.qmd`.
  New and revised content gets one div per subfile, a solution in its own `_sol` file included.
  Existing `_def-` style prefix names stay until a separate rename sweep.
  (Ezra, 2026-10-18; ai-config
  [`lecture-conventions`](https://github.com/Morrison-Lab/ai-config/blob/main/shared/writing/lecture-conventions.md)).
  rme's logistic-regression chapter is the bar: a 117-line spine over 95 short subfiles.
- **Check the rendered slides, not only the web page,** for boxed and coloured divs (2026-09-25;
  psw: [structure slides for the reader](https://morrison-lab.github.io/psw/chapters/teaching.html#structure-slides-for-the-reader)).
- **Credit adapted material in a `::: notes` div**, which keeps it off the slides (2026-09-25).
  psw's [credit other courses](https://morrison-lab.github.io/psw/chapters/teaching.html#credit-other-courses-without-summarizing-them) gives the general rule.
- **Do not put a lot of body text on section or title slides**
  (2026-09-29, [#271](https://github.com/Morrison-Lab/lds/issues/271)).
  In RevealJS, section and title slides (such as level-1 headings like `# Preface`)
  are vertically centered and do not scroll,
  so excess content gets clipped off the top and bottom of the screen.
  Use `{{< slidebreak >}}` after section headings and between large paragraphs
  to send body text onto subsequent content slides.
- **Put each video and other external media link beside the content it supports**
  (2026-10-02).
  The general rule is in
  [psw](https://github.com/Morrison-Lab/psw/blob/main/chapters/paper-organization.qmd).
  Here, put each video note in a `::: {.videos}` callout (titled "Watch")
  right after the exercise-and-solution pair, figure or definition that the video covers,
  so the videos are spread through the page.
  Put each link to a written tutorial or article, such as a GeeksforGeeks page,
  in a `::: {.reading}` callout (titled "Read") placed the same way,
  after the "Watch" callout when both sit at one spot (2026-10-05).
  Use a `*Media:*` note for other external material.
  Both callouts fold, open by default, and each has an id for links,
  added by `filters/watch-callouts.lua`: `watch-` or `read-` and a slug of the first of these that exists:
  the `vid-` id of an embedded video, the text of the first descriptive link,
  the YouTube id of a video link, the key of the first citation, the first paragraph or list item, or `box`.
  The page shows a § link in the title bar on hover.
  Because the slug follows the text, write the id yourself when other pages link to the box,
  as in `::: {.videos #watch-sanderson-backprop}`.
  Add `collapse="none"` to make one box unfoldable, or `collapse="true"` to start one folded.
  A consolidated list of a chapter's videos at its start or end may also appear,
  in addition to the callouts, never instead of them (Ezra, 2026-10-02).
  Ezra pointed to [the rmsc site](https://hbiostat.org/rmsc/) as an example of this placement.
  That site also embeds audio narration, which we do not have, so only external links appear here.
- **Generalization gets its own `#def-` div** (2026-09-25).
- **Name a book by its title where the book, not its authors, is the subject**:
  a bare `@key` renders as "Author [n]", such as "Hastie et al. [2]",
  so write "*The Elements of Statistical Learning* notes" with a bracketed citation.

## Writing rules kept elsewhere

These are general rules, with their full text in `Morrison-Lab/ai-config` (loaded as a Claude Code plugin)
or in [psw](https://github.com/Morrison-Lab/psw):

- [`definition-crossrefs`](https://github.com/Morrison-Lab/ai-config/blob/main/shared/writing/definition-crossrefs.md):
  link every mention of a defined term to its div, and define it before its first mention.
- [`informal-definitions`](https://github.com/Morrison-Lab/ai-config/blob/main/shared/writing/informal-definitions.md):
  a concept defined precisely in prose gets its own `#def-` div, one per term.
- [`hypothetical-examples`](https://github.com/Morrison-Lab/ai-config/blob/main/shared/writing/hypothetical-examples.md):
  prefer a real dataset to invented numbers.
- [`semantic-line-breaks`](https://github.com/Morrison-Lab/ai-config/blob/main/shared/writing/semantic-line-breaks.md):
  one phrase per source line, and do not reflow lines you are not changing.
- [`forward-references`](https://github.com/Morrison-Lab/ai-config/blob/main/shared/writing/forward-references.md):
  no "as we will see below".
- [`citations`](https://github.com/Morrison-Lab/ai-config/blob/main/shared/writing/citations.md):
  BibTeX keys with a chapter or page locator, and a page number only once checked against the source.
- [`quarto-revealjs-div-styling`](https://github.com/Morrison-Lab/ai-config/blob/main/shared/writing/quarto-revealjs-div-styling.md):
  slides box and colour every div, as the web page does.
- [Give each defined term its own div](https://morrison-lab.github.io/psw/chapters/teaching.html#give-each-defined-term-its-own-div),
  and never nest one theorem-type div inside another (Ezra, 2026-09-25):
  no `#def-`, `#thm-`, `#lem-`, `#cor-`, `#prp-`, `#cnj-`, `#exm-` or `#rem-` div inside any other.
  A special case, such as a linear evaluation function
  inside the definition of an evaluation function,
  gets its own div after the general one, which it links to.
- From psw (Ezra, 2026-09-25):
  - a bolded keyword in running prose usually marks an inline definition to move into a `#def-` div
    ([defining terms](https://morrison-lab.github.io/psw/chapters/defining-terms.html#guidelines-for-defining-terms));
  - start a sentence with its subject, not a dependent clause
    ([lead with the subject](https://morrison-lab.github.io/psw/chapters/conciseness.html#lead-with-the-subject));
  - name the referent instead of a bare demonstrative or "there", as in "getting there"
    ([demonstrative pronouns](https://morrison-lab.github.io/psw/chapters/grammar.html#demonstrative-pronouns));
  - never write "thing"; say which object, quantity or idea
    ([say which thing](https://morrison-lab.github.io/psw/chapters/word-choice.html#say-which-thing));
  - state the literal claim instead of an informal idiom
    ([vague and metaphorical language](https://morrison-lab.github.io/psw/chapters/word-choice.html#avoid-vague-and-metaphorical-language));
  - quote the primary source directly (Mitchell from Mitchell, not via Goodfellow et al.),
    and check the wording against the original
    ([quote the original](https://morrison-lab.github.io/psw/chapters/citations-evidence.html#quote-the-original));
  - title a section by its topic, and follow a quotation with a bare `[@key]` citation and no commentary
    ([content first](https://morrison-lab.github.io/psw/chapters/citations-evidence.html#put-the-content-first));
  - write our own version of another course's material rather than summarize it
    ([adapting material](https://morrison-lab.github.io/psw/chapters/citations-evidence.html#adapting-another-courses-material));
  - cut AI clichés: straw objections, personified abstractions, colon reveals,
    answered rhetorical questions and closing aphorisms
    ([AI tells](https://morrison-lab.github.io/psw/chapters/avoid-ai-tells.html)).
