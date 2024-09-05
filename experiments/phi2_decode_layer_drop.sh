CUDA_VISIBLE_DEVICES=0 python run_decode_layer_drop.py --n_drop=1 --cache_filename Phi2_similarity_BD_1.pt --model_save_path models/Phi2-BD1 & 
CUDA_VISIBLE_DEVICES=1 python run_decode_layer_drop.py --n_drop=2 --cache_filename Phi2_similarity_BD_2.pt --model_save_path models/Phi2-BD2 & 
CUDA_VISIBLE_DEVICES=2 python run_decode_layer_drop.py --n_drop=3 --cache_filename Phi2_similarity_BD_3.pt --model_save_path models/Phi2-BD3 & 
CUDA_VISIBLE_DEVICES=3 python run_decode_layer_drop.py --n_drop=4 --cache_filename Phi2_similarity_BD_4.pt --model_save_path models/Phi2-BD4 & 
CUDA_VISIBLE_DEVICES=4 python run_decode_layer_drop.py --n_drop=5 --cache_filename Phi2_similarity_BD_5.pt --model_save_path models/Phi2-BD5 & 
CUDA_VISIBLE_DEVICES=5 python run_decode_layer_drop.py --n_drop=6 --cache_filename Phi2_similarity_BD_6.pt --model_save_path models/Phi2-BD6 & 
CUDA_VISIBLE_DEVICES=6 python run_decode_layer_drop.py --n_drop=7 --cache_filename Phi2_similarity_BD_7.pt --model_save_path models/Phi2-BD7 & 
CUDA_VISIBLE_DEVICES=7 python run_decode_layer_drop.py --n_drop=8 --cache_filename Phi2_similarity_BD_8.pt --model_save_path models/Phi2-BD8 &