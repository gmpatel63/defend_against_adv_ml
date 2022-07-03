# directory names
DATA_DIR = 'model_input_data'
CSV_DIR = 'dataset_csvs'
OUTPUT_DIR_NAME = 'outputs'
INPUT_FNAME = 'input_data.csv'
MODEL_NAMES = ['cnn', 'context_aware', 'enhanced_context_aware', 'unfreezed_context_aware', 'srgan_context_aware', 'srgan_cnn']
CNN_DIR = 'cnn_model'
CONTEXT_AWARE_MODEL_DIR = 'context_aware_model'
ENHANCED_CONTEXT_AWARE_MODEL_DIR = 'enhanced_context_aware_model'
UNFREEZED_CONTEXT_AWARE_MODEL_DIR = 'unfreezed_context_aware_model'
SRGAN_CNN_DIR = 'srgan_cnn_model'
SRGAN_CONTEXT_AWARE_DIR = 'srgan_context_aware_model'
SRGAN_INPUT_DATA = 'srgan_input_data'
SRGAN_OUTPUT_DATA = 'srgan_output_data'

# constants
MRI_SHAPE = (172, 220, 156, 1)
OUTPUT_SHAPE = (1)
ANAT_FEAT_SHAPE = (132)

# training constants
TRAIN_PCT = 0.6
VALID_PCT = 0.2
TEST_PCT = 0.2
TF_DATASET_BUFFER = 48


ANATOMICAL_COLUMNS = ['4_3rd Ventricle',
 '11_4th Ventricle',
 '23_Right Accumbens Area',
 '30_Left Accumbens Area',
 '31_Right Amygdala',
 '32_Left Amygdala',
 '35_Brain Stem',
 '36_Right Caudate',
 '37_Left Caudate',
 '38_Right Cerebellum Exterior',
 '39_Left Cerebellum Exterior',
 '40_Right Cerebellum White Matter',
 '41_Left Cerebellum White Matter',
 '44_Right Cerebral White Matter',
 '45_Left Cerebral White Matter',
 '47_Right Hippocampus',
 '48_Left Hippocampus',
 '49_Right Inf Lat Vent',
 '50_Left Inf Lat Vent',
 '51_Right Lateral Ventricle',
 '52_Left Lateral Ventricle',
 '55_Right Pallidum',
 '56_Left Pallidum',
 '57_Right Putamen',
 '58_Left Putamen',
 '59_Right Thalamus Proper',
 '60_Left Thalamus Proper',
 '61_Right Ventral DC',
 '62_Left Ventral DC',
 '71_Cerebellar Vermal Lobules I-V',
 '72_Cerebellar Vermal Lobules VI-VII',
 '73_Cerebellar Vermal Lobules VIII-X',
 '75_Left Basal Forebrain',
 '76_Right Basal Forebrain',
 '100_Right ACgG anterior cingulate gyrus',
 '101_Left ACgG anterior cingulate gyrus',
 '102_Right AIns anterior insula',
 '103_Left AIns anterior insula',
 '104_Right AOrG anterior orbital gyrus',
 '105_Left AOrG anterior orbital gyrus',
 '106_Right AnG angular gyrus',
 '107_Left AnG angular gyrus',
 '108_Right Calc calcarine cortex',
 '109_Left Calc calcarine cortex',
 '112_Right CO central operculum',
 '113_Left CO central operculum',
 '114_Right Cun cuneus',
 '115_Left Cun cuneus',
 '116_Right Ent entorhinal area',
 '117_Left Ent entorhinal area',
 '118_Right FO frontal operculum',
 '119_Left FO frontal operculum',
 '120_Right FRP frontal pole',
 '121_Left FRP frontal pole',
 '122_Right FuG fusiform gyrus',
 '123_Left FuG fusiform gyrus',
 '124_Right GRe gyrus rectus',
 '125_Left GRe gyrus rectus',
 '128_Right IOG inferior occipital gyrus',
 '129_Left IOG inferior occipital gyrus',
 '132_Right ITG inferior temporal gyrus',
 '133_Left ITG inferior temporal gyrus',
 '134_Right LiG lingual gyrus',
 '135_Left LiG lingual gyrus',
 '136_Right LOrG lateral orbital gyrus',
 '137_Left LOrG lateral orbital gyrus',
 '138_Right MCgG middle cingulate gyrus',
 '139_Left MCgG middle cingulate gyrus',
 '140_Right MFC medial frontal cortex',
 '141_Left MFC medial frontal cortex',
 '142_Right MFG middle frontal gyrus',
 '143_Left MFG middle frontal gyrus',
 '144_Right MOG middle occipital gyrus',
 '145_Left MOG middle occipital gyrus',
 '146_Right MOrG medial orbital gyrus',
 '147_Left MOrG medial orbital gyrus',
 '148_Right MPoG postcentral gyrus medial segment',
 '149_Left MPoG postcentral gyrus medial segment',
 '150_Right MPrG precentral gyrus medial segment',
 '151_Left MPrG precentral gyrus medial segment',
 '152_Right MSFG superior frontal gyrus medial segment',
 '153_Left MSFG superior frontal gyrus medial segment',
 '154_Right MTG middle temporal gyrus',
 '155_Left MTG middle temporal gyrus',
 '156_Right OCP occipital pole',
 '157_Left OCP occipital pole',
 '160_Right OFuG occipital fusiform gyrus',
 '161_Left OFuG occipital fusiform gyrus',
 '162_Right OpIFG opercular part of the inferior frontal gyrus',
 '163_Left OpIFG opercular part of the inferior frontal gyrus',
 '164_Right OrIFG orbital part of the inferior frontal gyrus',
 '165_Left OrIFG orbital part of the inferior frontal gyrus',
 '166_Right PCgG posterior cingulate gyrus',
 '167_Left PCgG posterior cingulate gyrus',
 '168_Right PCu precuneus',
 '169_Left PCu precuneus',
 '170_Right PHG parahippocampal gyrus',
 '171_Left PHG parahippocampal gyrus',
 '172_Right PIns posterior insula',
 '173_Left PIns posterior insula',
 '174_Right PO parietal operculum',
 '175_Left PO parietal operculum',
 '176_Right PoG postcentral gyrus',
 '177_Left PoG postcentral gyrus',
 '178_Right POrG posterior orbital gyrus',
 '179_Left POrG posterior orbital gyrus',
 '180_Right PP planum polare',
 '181_Left PP planum polare',
 '182_Right PrG precentral gyrus',
 '183_Left PrG precentral gyrus',
 '184_Right PT planum temporale',
 '185_Left PT planum temporale',
 '186_Right SCA subcallosal area',
 '187_Left SCA subcallosal area',
 '190_Right SFG superior frontal gyrus',
 '191_Left SFG superior frontal gyrus',
 '192_Right SMC supplementary motor cortex',
 '193_Left SMC supplementary motor cortex',
 '194_Right SMG supramarginal gyrus',
 '195_Left SMG supramarginal gyrus',
 '196_Right SOG superior occipital gyrus',
 '197_Left SOG superior occipital gyrus',
 '198_Right SPL superior parietal lobule',
 '199_Left SPL superior parietal lobule',
 '200_Right STG superior temporal gyrus',
 '201_Left STG superior temporal gyrus',
 '202_Right TMP temporal pole',
 '203_Left TMP temporal pole',
 '204_Right TrIFG triangular part of the inferior frontal gyrus',
 '205_Left TrIFG triangular part of the inferior frontal gyrus',
 '206_Right TTG transverse temporal gyrus',
 '207_Left TTG transverse temporal gyrus']
