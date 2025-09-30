import os

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 20
SEQUENCE_LENGTH = 5
SAMPLING_RATE = 2

DATA_DIR = "../data"
RESULTS_DIR = "../results"
CHECKPOINTS_DIR = "../results/checkpoints"
VISUALIZATIONS_DIR = "../visualizations"
LOGS_DIR = "../logs"

TRAIN_IMAGES_PATH = os.path.join(DATA_DIR, "train_images.npy")
TRAIN_LABELS_PATH = os.path.join(DATA_DIR, "train_labels.npy")
TRAIN_FILENAMES_PATH = os.path.join(DATA_DIR, "train_filenames.npy")

VAL_IMAGES_PATH = os.path.join(DATA_DIR, "val_images.npy")
VAL_LABELS_PATH = os.path.join(DATA_DIR, "val_labels.npy")
VAL_FILENAMES_PATH = os.path.join(DATA_DIR, "val_filenames.npy")

TEST_IMAGES_PATH = os.path.join(DATA_DIR, "test_images.npy")
TEST_LABELS_PATH = os.path.join(DATA_DIR, "test_labels.npy")
TEST_FILENAMES_PATH = os.path.join(DATA_DIR, "test_filenames.npy")

UCF101_CLASSES = [
    'ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam',
    'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress',
    'Biking', 'Billiards', 'BlowDryHair', 'BlowingCandles', 'BodyWeightSquats',
    'Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke',
    'BrushingTeeth', 'CleanAndJerk', 'CliffDiving', 'CricketBowling',
    'CricketShot', 'CuttingInKitchen', 'Diving', 'Drumming', 'Fencing',
    'FieldHockeyPenalty', 'FloorGymnastics', 'FrisbeeCatch', 'FrontCrawl',
    'GolfSwing', 'Haircut', 'HammerThrow', 'Hammering', 'HandstandPushups',
    'HandstandWalking', 'HeadMassage', 'HighJump', 'HorseRace', 'HorseRiding',
    'HulaHoop', 'IceDancing', 'JavelinThrow', 'JugglingBalls', 'JumpRope',
    'JumpingJack', 'Kayaking', 'Knitting', 'LongJump', 'Lunges', 'MilitaryParade',
    'Mixing', 'MoppingFloor', 'Nunchucks', 'ParallelBars', 'PizzaTossing',
    'PlayingCello', 'PlayingDaf', 'PlayingDhol', 'PlayingFlute', 'PlayingGuitar',
    'PlayingPiano', 'PlayingSitar', 'PlayingTabla', 'PlayingViolin', 'PoleVault',
    'PommelHorse', 'PullUps', 'Punch', 'PushUps', 'Rafting', 'RockClimbingIndoor',
    'RopeClimbing', 'Rowing', 'SalsaSpin', 'ShavingBeard', 'Shotput',
    'SkateBoarding', 'Skiing', 'Skijet', 'SkyDiving', 'SoccerJuggling',
    'SoccerPenalty', 'StillRings', 'SumoWrestling', 'Surfing', 'Swing',
    'TableTennisShot', 'TaiChi', 'TennisSwing', 'ThrowDiscus',
    'TrampolineJumping', 'Typing', 'UnevenBars', 'VolleyballSpiking',
    'WalkingWithDog', 'WallPushups', 'WritingOnBoard', 'YoYo'
]

