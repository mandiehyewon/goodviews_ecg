import numpy as np
import torch

### augment_type ###
# 1: time shift
# 2: masking
# 3: amplitude scale


def augment(args, augment_type, x):
    leads = x.size(0) #12
    time_axis_length = x.size(1)

    time_mask_para = time_axis_length / 150     
    if augment_type == 1:  
        t_shift = int(np.random.uniform(low=args.tshift_min, high=args.tshift_max))
        for lead in range(leads):
            if t_shift >= 0:
                print (x[lead, t_shift:].size(), torch.zeros(t_shift).size())
                x[lead, :] = torch.cat([x[lead, t_shift:], torch.zeros(t_shift)], dim=0)
            else:
                x[lead, :] = torch.cat([torch.zeros(t_shift), x[lead, :t_shift]], dim=0)
            
    elif augment_type == 2:    
        for lead in range(leads):
            t_zero_masking = int(np.random.uniform(low=args.mask_min, high=args.mask_max))
            t_zero_masking_start = int(np.random.uniform(low=0, high=time_axis_length-t_zero_masking-1))
            x[lead, t_zero_masking_start:t_zero_masking_start+t_zero_masking] = 0

    elif augment_type == 3:    
        
        for lead in range(leads):
            amp_scale = round(float(np.random.uniform(low=args.amplitude_min, high=args.amplitude_max)),5)
            x[lead, :] = torch.mul(x[lead, :], amp_scale)

    elif augment_type == 4:    
        for lead in range(leads):
            gs_noise_factor = float(torch.random.uniform(low=args.noise_min, high=args.noise_max))
            gs_noise = torch.normal(mean=gs_noise_factor, std=0.01, size=torch.size(x[lead, :]))
            x[lead, :] = torch.add(x[lead, :], gs_noise)

    else:
        print("Error! select correct augmentation type")
        exit(1)
    
    return x
