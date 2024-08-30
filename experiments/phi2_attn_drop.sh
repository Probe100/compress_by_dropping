CUDA_VISIBLE_DEVICES=0 python run_attn_drop.py --model microsoft/phi-2 --n_drop=1 --cache_filename Phi2_similarity_AD_1.pt --model_save_path models/Phi2-AD1 & 
CUDA_VISIBLE_DEVICES=1 python run_attn_drop.py --model microsoft/phi-2 --n_drop=2 --cache_filename Phi2_similarity_AD_2.pt --model_save_path models/Phi2-AD2 & 
CUDA_VISIBLE_DEVICES=2 python run_attn_drop.py --model microsoft/phi-2 --n_drop=3 --cache_filename Phi2_similarity_AD_3.pt --model_save_path models/Phi2-AD3 & 
CUDA_VISIBLE_DEVICES=3 python run_attn_drop.py --model microsoft/phi-2 --n_drop=4 --cache_filename Phi2_similarity_AD_4.pt --model_save_path models/Phi2-AD4 & 
CUDA_VISIBLE_DEVICES=4 python run_attn_drop.py --model microsoft/phi-2 --n_drop=5 --cache_filename Phi2_similarity_AD_5.pt --model_save_path models/Phi2-AD5 & 
CUDA_VISIBLE_DEVICES=5 python run_attn_drop.py --model microsoft/phi-2 --n_drop=6 --cache_filename Phi2_similarity_AD_6.pt --model_save_path models/Phi2-AD6 & 
CUDA_VISIBLE_DEVICES=6 python run_attn_drop.py --model microsoft/phi-2 --n_drop=7 --cache_filename Phi2_similarity_AD_7.pt --model_save_path models/Phi2-AD7 & 
CUDA_VISIBLE_DEVICES=7 python run_attn_drop.py --model microsoft/phi-2 --n_drop=8 --cache_filename Phi2_similarity_AD_8.pt --model_save_path models/Phi2-AD8 &