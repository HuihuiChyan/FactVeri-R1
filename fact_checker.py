import argparse
import json
import os
import time
import re
from tqdm import tqdm
from typing import List, Dict, Optional
import logging
from concurrent.futures import ThreadPoolExecutor

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
)

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


# from search_api_searxng import SearchAPISearxng
from search_api import SearchAPI


from utils_prompts import (
    get_agentic_verification_instruction,
    get_baseline_verification_instruction,
    BEGIN_SEARCH_QUERY,
    END_SEARCH_QUERY,
    BEGIN_SEARCH_RESULT,
    END_SEARCH_RESULT,
)


def parse_args():

    parser = argparse.ArgumentParser(
        description="Run Agentic Fact-Checking using Search-O1 methodology."
    )

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input data file (JSONL format).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the output results.",
    )
    parser.add_argument(
        "--search_engine_url",
        type=str,
        default="http://localhost:8080",
        help="URL for the Searxng instance.",
    )
    parser.add_argument(
        "--search_api_type",
        type=str,
        default="serper",
        choices=("serper", "searxng"),
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="fact_checking_debug.log",
        help="Path to save the debug log file.",
    )  # 日志文件

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pre-trained VLLM model.",
    )

    parser.add_argument(
        "--max_search_limit",
        type=int,
        default=5,
        help="Maximum number of searches per claim.",
    )

    # 搜索并行
    parser.add_argument(
        "--search_concurrency",
        type=int,
        default=20,
        help="Maximum number of concurrent search requests.",
    )

    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature."
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Top-p sampling parameter."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens to generate per turn.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=None,
        help="Tensor parallel size for VLLM.",
    )

    # 需要评估
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation after generating results. Requires ground truth labels in the input file.",
    )

    # 基线模式
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run in baseline mode without search capabilities. Model relies only on internal knowledge.",
    )

    return parser.parse_args()


def extract_between(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    """
    从文本中提取被两个特殊标记包裹的内容。
    """
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def extract_final_verdict(final_output: str) -> str:
    """
    从模型的完整输出轨迹中提取最终结论（'Real' 或 'Not Real'）。
    此函数优先处理最后一次搜索结果之后的文本，以获取最终判断。
    搜索不区分大小写，并对周围的文本或Markdown标记具有鲁棒性。
    """
    # 专注于最后一次搜索注入后生成的文本
    last_segment = (
        final_output.split(END_SEARCH_RESULT)[-1]
        if END_SEARCH_RESULT in final_output
        else final_output
    )

    # # 清理文本：转为小写，移除星号，并去除首尾空格
    # clean_segment = last_segment.lower().replace("*", "").strip()

    if re.search(r"\<answer\>\s*Not Real\s*\</answer\>", last_segment):
        return "Unsupported"
    if re.search(r"\<answer\>\s*Real\s*\</answer\>", last_segment):
        return "Supported"

    return "Inconclusive"


def evaluate_final_results(results: List[Dict]):
    """
    通过将最终判断与输入文件中的真实标签进行比较，计算并打印评估指标。
    """
    y_true = []
    y_pred = []
    invalid_predictions = 0
    total_valid_items = 0

    # 将字符串标签映射为二进制值 (1 for supported, 0 for unsupported)
    # 这可以处理大小写变化。
    label_map = {"supported": 1, "unsupported": 0}

    for item in results:
        # 真实标签从 'label' 字段获取。
        true_label_str = item.get("label", "").lower()
        pred_label_str = item.get("final_verdict", "").lower()

        # 只要真实标签有效，就参与评估
        if true_label_str in label_map:
            total_valid_items += 1
            y_true.append(label_map[true_label_str])
            
            # 如果预测标签在mapping中
            if pred_label_str in label_map:
                y_pred.append(label_map[pred_label_str])
            else:
                # 预测标签不在mapping中，记录为解码混乱，并标记为错误
                invalid_predictions += 1
                # 将无效预测标记为与真实标签相反的值（确保错误）
                y_pred.append(1 - label_map[true_label_str])
        else:
            logging.warning(
                f"Skipping item for evaluation due to invalid ground truth label. "
                f"Ground Truth: '{item.get('label')}', Prediction: '{item.get('final_verdict')}'"
            )

    if not y_true:
        logging.error(
            "Evaluation could not be performed. No items with valid ground truth labels found."
        )
        return

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Calculate invalid prediction ratio
    invalid_ratio = invalid_predictions / total_valid_items if total_valid_items > 0 else 0

    # Print results to console
    print("\n--- Evaluation Results ---")
    print(f"Total items evaluated: {len(y_true)}")
    print(f"Invalid predictions: {invalid_predictions}")
    print(f"Invalid prediction ratio: {invalid_ratio:.4f} ({invalid_ratio*100:.2f}%)")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("--------------------------\n")

    # Optionally, return a dictionary of metrics
    metrics_dict = {
        "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(balanced_accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "evaluated_count": len(y_true),
        "invalid_predictions": invalid_predictions,
        "invalid_ratio": round(invalid_ratio, 4),
    }
    return metrics_dict


def main():
    args = parse_args()

    # 配置日志记录，输出到文件和控制台
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(args.log_file, mode="w"),  # 写入文件
            # logging.StreamHandler(),  # 输出到控制台
        ],
    )
    logging.info(f"Debug log will be saved to {args.log_file}")

    if args.tensor_parallel_size is None:
        args.tensor_parallel_size = torch.cuda.device_count()

    logging.info("Initializing model and tokenizer...")

    # 初始化VLLM模型
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        max_model_len=32768, # Change this according to your model
    )

    # 只在非基线模式下初始化搜索API
    if not args.baseline:
        logging.info("Initializing search API...")
        # 初始化搜索模块，缓存机制由其内部处理
        claim_searcher = SearchAPI(search_url=args.search_engine_url, search_type=args.search_api_type)
        # 确保缓存目录存在
        claim_searcher.cache_dict = claim_searcher.load_cache()
    else:
        logging.info("Running in baseline mode - search API disabled")
        claim_searcher = None

    logging.info(f"Loading data from {args.input_file}...")
    # 数据加载格式
    with open(args.input_file, "r") as f:
        input_data = [json.loads(x) for x in f.readlines() if x.strip()]

    # 准备初始输入和活动序列
    active_sequences = []
    for idx, item in enumerate(input_data):
        # 假设每个item中有一个'claim'字段或者需要被验证的'response'字段
        claim_to_verify = item.get("claim") or item.get("response")
        if not claim_to_verify:
            continue

        # 根据模式选择prompt模板
        if args.baseline:
            initial_prompt_str = get_baseline_verification_instruction(
                claim=claim_to_verify
            )
        else:
            initial_prompt_str = get_agentic_verification_instruction(
                claim=claim_to_verify
            )

        # 应用聊天模板
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": initial_prompt_str}],
            tokenize=False,
            add_generation_prompt=True,
        )

        # 记录初始的prompt
        logging.info(f"--- Preparing Item {idx+1} ---")
        logging.info(
            f"Initial Prompt for Item {idx+1}:\n{prompt}\n--------------------"
        )

        active_sequences.append(
            {
                "id": idx + 1,  # 添加ID用于日志追踪
                "original_item": item,
                "prompt": prompt,
                "output_history": [],
                "finished": False,
                "search_count": 0,
                "executed_search_queries": set(),
            }
        )

    # 主循环逻辑
    turn = 0
    pbar = tqdm(total=len(active_sequences), desc="Fact-Checking Progress")
    start_time = time.time()

    # 之所以这里的turn限制是max_search_limit + 1, 是因为当检测到turn=max_search_limit时，会不进行搜索而插入文本（让模型直接生成结果），
    # 之后，还需要在max_search_limit + 1 turn重新调用vllm生成结果
    while any(not seq["finished"] for seq in active_sequences) and turn <= args.max_search_limit + 1:

        sequences_needing_generation = [
            seq for seq in active_sequences if not seq["finished"]
        ]
        if not sequences_needing_generation:
            break

        # 控制台输出当前轮次信息
        print(
            f"\n--- Turn {turn}: Processing {len(sequences_needing_generation)} items ---"
        )

        # 准备VLLM输入
        prompts_to_generate = [seq["prompt"] for seq in sequences_needing_generation]

        sampling_params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            stop=[END_SEARCH_QUERY, tokenizer.eos_token],  # 关键停止条件
            include_stop_str_in_output=True,
        )

        # 生成
        vllm_outputs = llm.generate(prompts_to_generate, sampling_params)

        # 处理生成结果
        sequences_to_search = []
        for seq, output in zip(sequences_needing_generation, vllm_outputs):
            generated_text = output.outputs[0].text
            seq["prompt"] += generated_text
            seq["output_history"].append(generated_text)

            # 记录模型输出
            logging.info(f"--- Item {seq['id']} | Turn {turn} ---")
            logging.info(f"Model Generated:\n{generated_text}\n--------------------")

            # 在基线模式下，跳过搜索逻辑，直接标记为完成
            if args.baseline:
                logging.info(f"Item {seq['id']} finished generation (baseline mode).")
                seq["finished"] = True
                pbar.update(1)
            else:
                # 原有的搜索逻辑
                search_query = extract_between(
                    generated_text, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY
                )

                if search_query and turn < args.max_search_limit:
                    if search_query not in seq["executed_search_queries"]:
                        logging.info(
                            f"Queueing search for item {seq['id']}: '{search_query}'"
                        )
                        seq["executed_search_queries"].add(search_query)
                        # Add the sequence and its query to the list for concurrent processing
                        sequences_to_search.append((seq, search_query))
                    else:
                        logging.warning(
                            f"Skipping duplicate search for item {seq['id']}: '{search_query}'"
                        )
                        injection_text = f"\n{BEGIN_SEARCH_RESULT}\nYou have already searched for this. Please use previous results.\n{END_SEARCH_RESULT}\n"
                        seq["prompt"] += injection_text
                        seq["output_history"].append(injection_text)

                elif search_query and turn == args.max_search_limit:  # 搜索次数超限
                    logging.warning(
                        f"Search limit reached for item {seq['id']}. It has performed {seq['search_count']} searches."
                    )
                    injection_text = f"\n{BEGIN_SEARCH_RESULT}\nSearch limit reached. Please immediately make a decision based on the information you have.\n{END_SEARCH_RESULT}\n"
                    seq["prompt"] += injection_text
                    seq["output_history"].append(injection_text)
                else:  
                    # 没有搜索，认为已完成
                    # 或者仍然存在search_query，但是已经达到了max_search_limit + 1 turn，这种情况也强制结束
                    logging.info(f"Item {seq['id']} finished generation.")
                    seq["finished"] = True
                    pbar.update(1)

        # 并发搜索
        if sequences_to_search:
            logging.info(
                f"Executing {len(sequences_to_search)} searches concurrently (max workers: {args.search_concurrency})..."
            )

            # Prepare lists for the thread pool
            seq_list = [item[0] for item in sequences_to_search]
            query_list = [item[1] for item in sequences_to_search]

            # search_results_formatted = []
            # for query in query_list:
            #     search_result = claim_searcher.get_search_res(query)
            #     search_results_formatted.append(search_result)

            with ThreadPoolExecutor(max_workers=args.search_concurrency) as executor:
                # executor.map executes the searches in parallel and returns results in order
                search_results_formatted = list(
                    tqdm(
                        executor.map(claim_searcher.get_search_res, query_list),
                        total=len(query_list),
                        desc="Concurrent Searches",
                    )
                )

            # Process the results and inject them back into the prompts
            # The search APIs now return formatted strings directly
            for seq, formatted_snippets in zip(seq_list, search_results_formatted):
                logging.info(
                    f"Injecting Search Results for item {seq['id']}:\n{formatted_snippets}\n--------------------"
                )
                injection_text = f"\n{BEGIN_SEARCH_RESULT}\n{formatted_snippets}\n{END_SEARCH_RESULT}\n"
                seq["prompt"] += injection_text
                seq["output_history"].append(injection_text)

        turn += 1

    pbar.close()
    total_time = time.time() - start_time
    logging.info(f"\nProcessing finished in {total_time:.2f} seconds.")

    # 保存最终结果
    final_results = []
    for seq in active_sequences:
        # 提取最终结论
        final_verdict = extract_final_verdict("".join(seq["output_history"]))

        result_item = seq["original_item"].copy()
        result_item["fact_checking_trace"] = seq["output_history"]
        result_item["final_verdict"] = final_verdict
        result_item["search_queries_made"] = list(seq["executed_search_queries"])
        final_results.append(result_item)

    # Run evaluation if the flag is set
    if args.evaluate:
        logging.info("Running evaluation on the generated results...")
        evaluate_final_results(final_results)

    # 最后保存一次缓存 (仅在非基线模式下)
    if not args.baseline and claim_searcher:
        claim_searcher.save_cache()

    # 创建父目录（如果不存在）
    output_dir = os.path.dirname(args.output_file)
    if output_dir:  # 非空路径才需要创建目录
        os.makedirs(output_dir, exist_ok=True)
    # 写入文件
    with open(args.output_file, "w", encoding="utf-8") as f:
        for item in final_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logging.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
