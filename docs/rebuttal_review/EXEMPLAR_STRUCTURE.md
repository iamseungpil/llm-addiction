# The structure every reviewer section must have

Taken from a NeurIPS author response the authors judged exemplary (Submission 16618). Copy the
shape, not the content.

## 1. Opening (before any [Weakness N] block)

Three parts, in this order.

**(a) One sentence of thanks that says what the reviewer did.** Not boilerplate.
> "We thank the reviewer for reading our paper so closely, and for pressing on the places where
> our evidence was thinnest."

**(b) One sentence naming what the reviewer's points have in common**, so the reader sees the
shape of the section before entering it.
> "Both points ask whether our result is narrower than we claimed --- narrower in domain, and
> narrower in the choice of distribution --- and both were fair to raise. We ran new experiments
> for each. Here is where they landed."

**(c) A bulleted block, one bullet per reviewer concern, in the reviewer's own terms.** Each
bullet is a **bold question**, then the answer in the first few words, then at most two sentences
of substance. This is the single most important element: a reviewer who reads only this block
must already know our answer to every point.
> - **Could a properly tuned temperature do what our method does?** No. We searched temperature
>   seriously this time --- five constant settings and two annealing schedules --- and none of
>   them keeps up once several samples are drawn.
> - **Are the accuracy gains real, or seed noise?** Real, though modest where we said they were
>   modest. Five seeds reproduce the numbers in the paper.
> - **Does anything separate learned content from injection itself?** Not yet. We agree this is
>   the decisive experiment. We are running the design the reviewer proposed and will post the
>   result in this thread.

Note the answers lead with a verdict word: "No." / "Real, though modest" / "Not yet." /
"Partly, and the boundary turns out to be informative." / "Yes, and we no longer need to hold it
fixed."

**(d) One closing sentence, identical wording in all three sections:**
> "If any part of our response falls short, we would be glad to take it further during the
> discussion period."

## 2. Inside each [Weakness N] / [Question M] block

- Merge a weakness and the question that restates it into one block, headed
  `[Weakness N & Question M]`.
- Lead with the verdict, then the evidence.
- Cross-reference rather than repeat: "Rebuttal Table 5 (in our response to Reviewer SQDD)".
- Reframe a concession so it supports the claim rather than only conceding:
  > "We are not arguing that random injection beats trained soft prompts, but that it reaches
  > comparable accuracy without any training. ... That is the shape our claim predicts."
- Defuse an attack before it is made. Example: a zero standard deviation looks suspicious, so
  they explain it:
  > "AIME24 has thirty problems, so accuracy moves in steps of 3.33 points; the 16.67 +- 0.00
  > entry means all five seeds solved the same number."
- For an unfinished experiment, say so plainly and say when:
  > "Not yet. We agree this is the decisive experiment. We are running the design the reviewer
  > proposed and will post the result in this thread."

## 3. What the exemplar never does

- No "We sincerely thank Reviewer X for the thoughtful review and encouraging assessment."
  (It appears in their draft, commented out, replaced by the specific version.)
- No paragraph that restates a table the reader can see.
- No admission of a fault the paper does not actually have.
