To implement the extended evaluation on the Fire Segmentation task, you have to set up the Official Implementation of the NAS-Unet (https://github.com/tianbaochou/NasUnet).
Please set up the NAS-Unet source code by following their instruction, and update their repository by using the provided files to evaluate our proposed FireSegNASUnet architecture.
1. Download the required dataset and set up the NasUnet-master directory.
2. Copy-paste the "flame1.py" file in the NasUnet-master->util->datasets folder.
3. Copy-paste the file "nas_firenet.yml" in the NasUnet-master->configs->nas_unet folder.
4. Copy-paste the file "train.py" into the NasUnet-master->experiment directory.
5. Update the "geno_searched.py" file in NasUnet-master->models by adding the discovered architecture genotype.

