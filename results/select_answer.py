import pdb
import json
with open("hotpotqa_subset_judged_Qwen2.5-7B-Instruct_agentic.jsonl", "r") as fin1,\
open("hotpotqa_subset_judged_Qwen2.5-7B-Instruct_baseline.jsonl", "r") as fin2:
    lines1 = [json.loads(line.strip()) for line in fin1.readlines()]
    lines2 = [json.loads(line.strip()) for line in fin2.readlines()]
    error_count = 0
    for line1, line2 in zip(lines1, lines2):
        # if line1["final_verdict"].lower() == line1['label'] and \
        #     line2["final_verdict"].lower() != line2['label']:
        #     print(json.dumps(line1['fact_checking_trace'], indent=4))
        #     print(json.dumps(line2['fact_checking_trace'], indent=4))
        #     import pdb;pdb.set_trace()
        if line1["final_verdict"].lower() != line1['label']:
            print(json.dumps(line1['fact_checking_trace'], indent=4))
            import pdb;pdb.set_trace()
        print(f"Final error count {error_count}")