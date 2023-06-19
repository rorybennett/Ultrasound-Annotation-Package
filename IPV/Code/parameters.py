'''
Parameters used in: Creating the datasets and Training the Model. Other scripts use the values here but the main ones
 are those 2.

Made some changes to how they represent strings. Didn't like their method. May have to monitor 'File Not Found' errors.

Also changed path naming to match parameter values, not sure why they didn't do that.
'''

scan_type = 'transverse'
# scan_type = 'sagittal'
# Number of folds to create, default was 10.
num_of_folds = 1
# Number of points: 4 for transverse, 2 for sagittal.

# ======================================================================================================================
# Data paths
# ======================================================================================================================
fold_lists_path = '../DATA/FOLD_LISTS'
# ======================================================================================================================

# ======================================================================================================================
# Data
# ======================================================================================================================
sub_patches = True
sub_patch_scales = [64, 128, 256, 512]
phase = 'Test'
# Number of patches in a training image, default was 200.
patches_per_sample = 200
# Step size when choosing test patch centres, default was 10.
test_data_step = 10
tasks_classes = [
    [(0, 15), (15, 25), (25, 40), (40, 60), (60, 85), (85, 115), (115, 150), (150, 190), (190, 235), (235, 285),
     (285, 1000)],
    [(0, 45), (45, 90), (90, 135), (135, 180), (180, 225), (225, 270), (270, 315), (315, 360)]
]
# num_of_classes = [len(l) for p in range(num_of_pts) for l in tasks_classes]
points = 'doctor'
# ======================================================================================================================

# ======================================================================================================================
# Train
# ======================================================================================================================
train_path = f'../Code_Data/{num_of_folds}FOLDS/{scan_type.upper()}/' \
             f'doctor_{sub_patch_scales}_{patches_per_sample}persample'
batch_size = 8
# Number of training epochs, default was 15.
training_epochs = 8
# Learning rate, default was 0.01.
lr = 0.01
lr_schedule = True
train_network = 'resnet18_pt'
loss_print = 1600 // batch_size
# ======================================================================================================================

# ======================================================================================================================
# Map Creation
# ======================================================================================================================
map_network = 'resnet18_pt'
path = f'../Code_Data/{num_of_folds}FOLDS/{scan_type.upper()}/doctor_{sub_patch_scales}_{patches_per_sample}persample'
map_creation_path = f'{path}/TestResults'
# I think they use epoch 5 as the validation accuracy had saturated by now.
epoch = 5
model_paths = []
for i in range(10):
    model_paths.append(f'{path}/resnet18_pt_bs8_curr_state_f{i + 1}_lr{lr}_e{epoch}.pth')
weighted_map = True
layers_to_vote = 3  # 1 or 3
# ======================================================================================================================
