## Human Detection - train
#python main.py --epochs 200 --output_dir weights/base --box_feature 16 
#python main.py --epochs 200 --output_dir weights/base_vcn --box_feature 16 

#python main.py --batch_size 128 --epochs 50 --pose hmdr --output_dir weights/base_vc31_hmdr/ --frozen_weights weights/base_vcn31/checkpoint0199.pth --box_feature 16 --device cuda:1 

python main.py --batch_size 8 --pose hmdr --eval --output_dir weights/mpii_hmdr/ --resume weights/rpet.pth
#python main.py --batch_size 8 --pose hmdr --eval --output_dir weights/mpii_vcn31_hmdr/ --resume weights/rpet_vcn.pth --box_feature 16

