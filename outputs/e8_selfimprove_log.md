# E8 - Self-improving prompt log

LLM: `extractive`  |  dev = 18 questions, held-out test = 18 questions
Objective: `0.7*accuracy + 0.3*faithfulness - 0.0005*input_tokens`

> Note: the extractive fallback ignores prompt wording, so instruction edits cannot change its answers; only token cost changes. Run with an Ollama / OpenAI-compatible model to see real gains.

|   iter | change                                                                | accepted   |   dev_accuracy |   dev_faithfulness |   dev_input_tokens |   objective |
|-------:|:----------------------------------------------------------------------|:-----------|---------------:|-------------------:|-------------------:|------------:|
|      0 | baseline (default Jinja2 instruction, no demos)                       | yes        |          0.944 |              1.000 |            378.167 |       0.772 |
|      1 | failure-driven edit: +'text' rule                                     | no         |          0.944 |              1.000 |            397.167 |       0.762 |
|      1 | bootstrapped few-shot: +2 demos from correct dev answers (Q018, Q033) | no         |          0.944 |              1.000 |            620.167 |       0.651 |
|      2 | failure-driven edit: +'brevity' rule                                  | no         |          0.944 |              1.000 |            385.167 |       0.768 |
|      3 | failure-driven edit: +'table' rule                                    | no         |          0.944 |              1.000 |            415.167 |       0.753 |
|      4 | failure-driven edit: +'image' rule                                    | no         |          0.944 |              1.000 |            404.167 |       0.759 |

## Instruction per iteration

- iter 0 (accepted): `Answer the question precisely and briefly.` demos=[]
- iter 1 (rejected): `Answer the question precisely and briefly. Copy identifiers, versions and numbers exactly as written in the context.` demos=[]
- iter 1 (rejected): `Answer the question precisely and briefly.` demos=['Q018', 'Q033']
- iter 2 (rejected): `Answer the question precisely and briefly. Answer in at most 20 words.` demos=[]
- iter 3 (rejected): `Answer the question precisely and briefly. For table evidence, find the row whose first cell matches the identifier in the question and return the value of the column the question asks for, with its unit.` demos=[]
- iter 4 (rejected): `Answer the question precisely and briefly. Values from figures appear after 'Text in image:'; copy the value next to the label that the question names.` demos=[]

## Held-out test split

| prompt    |   answer_accuracy |   faithfulness |   avg_input_tokens |   objective |
|:----------|------------------:|---------------:|-------------------:|------------:|
| initial   |             0.778 |          1.000 |            387.944 |       0.650 |
| optimised |             0.778 |          1.000 |            387.944 |       0.650 |

## Final prompt

Instruction: `Answer the question precisely and briefly.`
Demos: []