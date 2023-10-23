dataset=$1

for shift in gaussian_noise gaussian_blur salt_pepper_noise uniform_noise image_transform pixel_shuffle pixel_dropout;
do
for hidden in 100 650;
do
for layers in 1 3;
do
for run in 0 1 2 3 4;
do
for intensity in 0 1 2 3 4 5;
do
for method in topological_var;
do
sbatch --export=hidden=$hidden,layers=$layers,run=$run,intensity=$intensity,dataset=$dataset,method=$method,shift=$shift make_witness_vectors.batch;
done;
done;
done;
done;
done;
done;
