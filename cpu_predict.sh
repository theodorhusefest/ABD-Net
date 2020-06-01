python3 classify.py -t pedestrianreid -s not_used \
        --load-weights=checkpoints/final_model.pth.zip.tar \
        --root=data/milestone3 \
        --save-dir=test \
        --video=video_1 \
        --csv=data/milestone3/video_1/yolov3_outputs/query_list.csv