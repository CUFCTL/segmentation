# Train script to loop through all 31 levels of compression
# Manually delete the compression_weights folder before you run this script
# But a better option would be to create compression_weights directory and append the current date/timestamp to it
mkdir weights/compression_weights
for i in 5 10 15 20 25 30; do
    python train.py configs/rn18_pyramid_rellis_compression.py --store_dir=weights --cr $i
    mv weights/cr_$i* weights/compression_weights
done