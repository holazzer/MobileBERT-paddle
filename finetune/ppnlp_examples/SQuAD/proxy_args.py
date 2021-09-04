class ProxyArgs:
    model_type = "bert"
    model_name_or_path = "bert-base-uncased"
    max_seq_length = 384
    batch_size = 12
    learning_rate = 3e-5
    num_train_epochs = 2
    logging_steps = 1000
    warmup_proportion = 0.1
    weight_decay = 0.01
    output_dir = "./tmp/squad/"
    device_= "gpu"
    do_train = True
    do_predict = True
    adam_epsilon = 1e-8
    max_grad_norm = 1.0
    max_steps = -1
    save_steps = 500
    seed = 42
    doc_stride = 128
    n_best_size = 20
    null_score_diff_threshold = 0.0
    max_query_length = 64
    max_answer_length = 30
    do_lower_case = True
    version_2_with_negative = False
    verbose = True







