### augment_type ###
# 1: time shift
# 2: masking
# 3: amplitude scale


def Augment(args, features):
    augment_type = random.randint(1,3)
    fs = float(args.sample_rate)

    feature = features[0]
    channel_num = feature.size(0) #12
    time_axis_length = feature.size(1)

    time_mask_para = time_axis_length / 150     
    if augment_type == 1:  
        t_shift = int(np.random.uniform(low=args.time_shift_min, high=args.time_shift_max))  
        for ch in range(channel_num):
            if t_shift >= 0:
                feature[ch, :] = torch.cat([feature[ch, t_shift:], torch.zeros(t_shift)], dim=0)
            else:
                feature[ch, :] = torch.cat([torch.zeros(t_shift), feature[ch, :t_shift]], dim=0)
            
    elif augment_type == 2:    
        for ch in range(channel_num):
            t_zero_masking = int(np.random.uniform(low=args.zero_masking_min, high=args.zero_masking_max))
            t_zero_masking_start = int(np.random.uniform(low=0, high=time_axis_length-t_zero_masking-1))
            feature[ch, t_zero_masking_start:t_zero_masking_start+t_zero_masking] = 0

    elif augment_type == 3:    
        for ch in range(channel_num):
            amp_scale = round(float(np.random.uniform(low=args.amplitude_min, high=args.amplitude_max)),5)
            feature[ch, :] = torch.mul(feature[ch, :], amp_scale)

    else:
        print("Error! select correct augmentation type")
        exit(1)
    
    return (feature, features[1], features[2])