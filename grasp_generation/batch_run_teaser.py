import os

def main():
    
    os.system(f"python3 main_seq.py --batch_size 128 \
                                    --name teaser_0 \
                                    --specific_os 3 \
                                    --object_code_list ddg-ycb_013_apple \
                                    --scale 0.045 \
                                    ")
    # os.system(f"python isaac_test.py --eval_dir ../data/experiments/teaser_0/")
    
    # os.system(f"python3 main_seq.py --batch_size 128 \
    #                                 --name teaser_1 \
    #                                 --specific_os 1 \
    #                                 --object_code_list ddg-ycb_011_banana \
    #                                 --scale 0.135 \
    #                                 --prev_grasp_path_list ../data/experiments/teaser_0/validated/0_ddg-ycb_013_apple.npy \
    #                                 ")
    # os.system(f"python isaac_test.py --eval_dir ../data/experiments/teaser_1/")
    
    # os.system(f"python3 main_seq.py --batch_size 128 \
    #                                 --name teaser_2 \
    #                                 --specific_os 6 \
    #                                 --object_code_list ddg-ycb_012_strawberry \
    #                                 --scale 0.035 \
    #                                 --prev_grasp_path_list ../data/experiments/teaser_1/validated/0_ddg-ycb_011_banana.npy \
    #                                 ")
    # os.system(f"python isaac_test.py --eval_dir ../data/experiments/teaser_2/")
    
if __name__ == "__main__":
    main()
