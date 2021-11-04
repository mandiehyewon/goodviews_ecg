def build_contrastive_batch(batch_patients):
    
"""
param batch_patients: list of collections of ECGs per patient. batch_patients[i] contains all ECGs for patient i
returns: list of ECGs to use as input to constrastive model of length 2* number of patients given 

"""

batch_half_1 = []
batch_half_2 = []

for i in range(len(batch_patients)):
    pt_ecg_1 = random.choice(batch_patients[i])
    pt_ecg_2 = random.choice(batch_patients[i]) # choosing with replacement, so could be the same as pt_ecg_1
    
    if gaussian:
        pt_ecg_1 = obtain_perturbed_frame(pt_ecg_1)
        pt_ecg_2 = obtain_perturbed_frame(pt_ecg_2)
    
    pt_ecg_1 = normalize_frame(pt_ecg_1)
    pt_ecg_2 = normalize_frame(pt_ecg_2)
        
    batch_half_1.append(pt_ecg_1)
    batch_half_2.append(pt_ecg_2)

    return batch_half_1 + batch_half_2

def obtain_perturbed_frame(frame):
    """ Apply Gaussian Noise to Frame 
    Args:
        frame (numpy array): frame containing ECG data
    Outputs
        frame (numpy array): perturbed frame based
    """
    variance_factor=10
    gauss_noise = np.random.normal(0,variance_factor,size=frame_size)
    frame = frame + gauss_noise

    return frame

def normalize_frame(self,frame):
    if isinstance(frame,np.ndarray):
        frame = (frame - np.min(frame))/(np.max(frame) - np.min(frame) + 1e-8)
    elif isinstance(frame,torch.Tensor):
        frame = (frame - torch.min(frame))/(torch.max(frame) - torch.min(frame) + 1e-8)
    return frame
