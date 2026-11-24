# E5 - Failure analysis

## Q002 (text) - cause: `wrong_evidence`
- question: Which defect ID is fixed in O-DU build R24.5.325?
- gold: DEF-87139 (release_note_325.pdf p.1, gold rank 2)
- system: Issue ID for ISS-325-5 is ISS-325-5 [4].
- proposed fix: The right chunk was retrieved but the answer used another unit (caption, other row or other modality). Strengthen modality routing, compress context harder, or add a table/figure few-shot demo to the prompt.

## Q003 (text) - cause: `wrong_evidence`
- question: Which defect ID is fixed in O-DU build R25.3.763?
- gold: DEF-79228 (release_note_763.pdf p.1, gold rank 4)
- system: Issue ID for ISS-763-1 is ISS-763-1 [2].
- proposed fix: The right chunk was retrieved but the answer used another unit (caption, other row or other modality). Strengthen modality routing, compress context harder, or add a table/figure few-shot demo to the prompt.

## Q014 (table) - cause: `wrong_evidence`
- question: How much memory is allocated to the Slice Assurance xApp in Region-396?
- gold: 4 GB (ric_arch_region396.pdf p.2, gold rank 1)
- system: Data Collection In Region-396 the Slice Assurance xApp subscribes to E2SM-KPM reports every 500 ms [3].
- proposed fix: The right chunk was retrieved but the answer used another unit (caption, other row or other modality). Strengthen modality routing, compress context harder, or add a table/figure few-shot demo to the prompt.

## Q029 (image) - cause: `generation_error`
- question: Which worst hour is annotated in the handover trend chart of Cluster-801?
- gold: 20:00 (ho_kpi_cluster801.pdf p.3, gold rank 1)
- system: HOSuccess Rate-Cluster-801 [1].
- proposed fix: Context contained the value but the generator paraphrased or picked a neighbouring value. Add an instruction to copy values with units exactly; use the optimised prompt.

## Q032 (image) - cause: `generation_error`
- question: What T3560 timer value is shown in the registration call flow of Lab-989?
- gold: 6 s (core_pdu_lab989.pptx p.4, gold rank 1)
- system: 1 Registration Request [1].
- proposed fix: Context contained the value but the generator paraphrased or picked a neighbouring value. Add an instruction to copy values with units exactly; use the optimised prompt.
