import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data import IndexingTrainDataset, QueryEvalCollator

def calculate_mrr(
    model,
    tokenizer,
    test_dataset,
    restrict_decode_vocab_fn,
    batch_size=64,
    max_length=20, # Độ dài tối đa khi generate, giống trong QueryEvalCallback
    num_beams=10, # Số beam, giống trong QueryEvalCallback
    num_return_sequences=10 # Số sequence trả về, giống trong QueryEvalCallback
):
    """
    Tính toán chỉ số Mean Reciprocal Rank (MRR) cho mô hình trên test_dataset.
    """
    model.eval() # Chuyển mô hình sang chế độ đánh giá

    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=QueryEvalCollator(tokenizer, padding="longest"),
        shuffle=False,
        drop_last=False,
        num_workers=4, # Có thể điều chỉnh số worker
    )

    total_reciprocal_rank = 0.0
    num_queries = len(test_dataset)

    print(f"Bắt đầu tính MRR trên {num_queries} truy vấn...")

    for batch in tqdm(dataloader, desc="Tính MRR"):
        inputs, labels = batch # labels ở đây là các docid đúng dạng chuỗi

        with torch.no_grad():
            # Generate các ứng viên docid sử dụng beam search
            batch_beams = model.generate(
                inputs["input_ids"].to(model.device),
                max_length=max_length,
                num_beams=num_beams,
                prefix_allowed_tokens_fn=restrict_decode_vocab_fn, # Sử dụng hàm giới hạn từ vựng
                num_return_sequences=num_return_sequences,
                early_stopping=True, # Dừng sớm nếu tìm thấy EOS token
                # Các tham số khác có thể cần tùy vào cách generate của T5
                # Ví dụ: no_repeat_ngram_size=...
            ).reshape(inputs["input_ids"].shape[0], num_return_sequences, -1)

            # Xử lý từng truy vấn trong batch
            for beams, label in zip(batch_beams, labels):
                # Giải mã các beam search results thành chuỗi văn bản (docid string)
                rank_list = tokenizer.batch_decode(beams, skip_special_tokens=True)
                print(rank_list)
                # Tìm vị trí (hạng) của docid đúng trong danh sách kết quả (rank_list)
                # rank_list là danh sách 0-based index
                try:
                    # Tìm index đầu tiên của label trong rank_list (chỉ xét 10 kết quả)
                    correct_rank_index = rank_list[:num_return_sequences].index(label)
                    # Hạng (rank) là index + 1
                    rank = correct_rank_index + 1
                    # Tính reciprocal rank
                    reciprocal_rank = 1.0 / rank
                    total_reciprocal_rank += reciprocal_rank
                except ValueError:
                    # Nếu label không có trong 10 kết quả hàng đầu
                    reciprocal_rank = 0.0
                    # total_reciprocal_rank không thay đổi

    # Tính Mean Reciprocal Rank
    mrr = total_reciprocal_rank / num_queries if num_queries > 0 else 0.0

    return mrr

# --- Phần chính để chạy ---
if __name__ == "__main__":
    # Thông tin checkpoint
    repo_id = "ngocnamk3er/dsi_code_t5_base_kaggle_6_6"
    commit_hash = "85f1cdc09e8e471e64e55c822b8b00b7945d636c" # Hash của checkpoint bạn muốn tải

    # Cấu hình
    model_name_base = "Salesforce/codet5-base" # Tên mô hình gốc để tải cấu hình nếu cần
    cache_dir = "cache"
    L = 32 # Max length tài liệu, giống khi train
    test_data_path = "./Vault_valid_python_clean.json" # Đường dẫn đến tập dữ liệu test

    # 1. Tải Tokenizer và Model từ checkpoint cụ thể
    print(f"Đang tải tokenizer từ {repo_id} revision {commit_hash}...")
    tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=commit_hash, cache_dir=cache_dir)

    print(f"Đang tải model từ {repo_id} revision {commit_hash}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(repo_id, revision=commit_hash, cache_dir=cache_dir)

    # Đẩy model lên GPU nếu có
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Sử dụng thiết bị: {device}")

    # 2. Tải tập dữ liệu test
    print(f"Đang tải tập dữ liệu test từ {test_data_path}...")
    test_dataset = IndexingTrainDataset(
        path_to_data=test_data_path,
        max_length=L,
        cache_dir=cache_dir,
        tokenizer=tokenizer, # Sử dụng tokenizer đã tải
    )

    # 3. Định nghĩa lại hàm giới hạn từ vựng giải mã (cần tokenizer đã tải)
    # Copy lại logic từ hàm main gốc của bạn
    SPIECE_UNDERLINE = " "
    INT_TOKEN_IDS = []
    for token, id in tokenizer.get_vocab().items():
        if token[0] == SPIECE_UNDERLINE:
            if token[1:].isdigit():
                INT_TOKEN_IDS.append(id)
        if token == SPIECE_UNDERLINE:
            INT_TOKEN_IDS.append(id)
        elif token.isdigit():
            INT_TOKEN_IDS.append(id)
    INT_TOKEN_IDS.append(tokenizer.eos_token_id)

    def restrict_decode_vocab(batch_idx, prefix_beam):
        return INT_TOKEN_IDS

    # 4. Tính toán MRR
    mrr_score = calculate_mrr(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        restrict_decode_vocab_fn=restrict_decode_vocab,
        batch_size=64, # Sử dụng batch size phù hợp với VRAM của bạn
    )

    print(f"\nMRR @10 cho checkpoint {commit_hash}: {mrr_score:.4f}")