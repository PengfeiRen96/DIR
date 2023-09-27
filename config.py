class Config:
    phase = 'train'
    pre = 'DIR'
    stage_num = 3
    dataset = 'IntagHand'
    output_root = './checkpoint/DIR-PosEmb-Scale' # set output root
    data_dir = './data/interhand2.6m/'
    mano_path = './misc/mano'
    checkpoint = '' # For continue train
    continue_train = False

    lr = 3e-4
    lr_scheduler = 'cosine' # cosine, step
    total_epoch = 50

    # network
    backbone = 'resnet50'
    num_vert = 778
    joint_num = 21
    pretrain_dir = './pretrain/mscan_s.pth'

    # training
    input_img_shape = (256, 256)
    batch_size = 64
    num_worker = 16

    # -------------
    save_epoch = 1
    eval_interval = 1
    print_iter = 100
    draw_iter = 100
    num_epoch_to_eval = 80

    vis = False
    # -------------
    experiment_name = pre + '_{}'.format(backbone)

cfg = Config()
