python crossmodal_transfer.py --dataset=imagenet --training_modality=image --gap_method=original --n_epochs=100
python crossmodal_transfer.py --dataset=imagenet --training_modality=text --gap_method=original --n_epochs=100
python crossmodal_transfer.py --dataset=imagenet --training_modality=image --gap_method=centering --n_epochs=100
python crossmodal_transfer.py --dataset=imagenet --training_modality=text --gap_method=centering --n_epochs=100

python crossmodal_transfer.py --dataset=coco --training_modality=image --gap_method=original --n_epochs=25
python crossmodal_transfer.py --dataset=coco --training_modality=text --gap_method=original --n_epochs=25
python crossmodal_transfer.py --dataset=coco --training_modality=image --gap_method=centering --n_epochs=25
python crossmodal_transfer.py --dataset=coco --training_modality=text --gap_method=centering --n_epochs=25