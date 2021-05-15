import tensorflow_cloud as tfc

tfc.run(
    requirements_txt="/home/pafrank/Documents/coding/chess-position-embedding/requirements.txt",
    entry_point="/home/pafrank/Documents/coding/chess-position-embedding/chesspos/demo/train_model.py",
    distribution_strategy='auto',
    chief_config=tfc.COMMON_MACHINE_CONFIGS['K80_1X'],
    worker_count=0
)