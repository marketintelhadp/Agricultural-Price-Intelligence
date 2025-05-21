import os
from keras.metrics import MeanSquaredError

# config.py

CONFIG = {
    "Azadpur": {
        "Cherry": {
            "Makhmali": {
                "Fancy": {
                    "model": "models/Azadpur/Makhmali/Fancy/lstm_Makhmali_grade_Fancy.h5",
                    "dataset": "data/raw/processed/Azadpur/Makhmali_Fancy_dataset.csv"
                },
                "Special": {
                    "model": "models/Azadpur/Makhmali/Special/lstm_Makhmali_grade_Special.h5",
                    "dataset": "data/raw/processed/Azadpur/Makhmali_Special_dataset.csv"
                },
                "Super": {
                    "model": "models/Azadpur/Makhmali/Super/lstm_Makhmali_grade_Super.h5",
                    "dataset": "data/raw/processed/Azadpur/Makhmali_Super_dataset.csv"
                }
            },
            "Misri": {
                "Fancy": {
                    "model": "models/Azadpur/Misri/Fancy/lstm_Misri_grade_Fancy.h5",
                    "dataset": "data/raw/processed/Azadpur/Misri_Fancy_dataset.csv"
                },
                "Special": {
                    "model": "models/Azadpur/Misri/Special/lstm_Misri_grade_Special.h5",
                    "dataset": "data/raw/processed/Azadpur/Misri_Special_dataset.csv"
                },
                "Super": {
                    "model": "models/Azadpur/Misri/Super/lstm_Misri_grade_Super.h5",
                    "dataset": "data/raw/processed/Azadpur/Misri_Super_dataset.csv"
                }
            }
        }
    },
    "Ganderbal": {
        "Cherry": {
            "Cherry": {
                "Large": {
                    "model": "models/Ganderbal/Cherry/Large/lstm_Cherry_grade_Large.h5",
                    "dataset": "data/raw/processed/Ganderbal/Cherry_Large_dataset.csv"
                },
                "Medium": {
                    "model": "models/Ganderbal/Cherry/Medium/lstm_Cherry_grade_Medium.h5",
                    "dataset": "data/raw/processed/Ganderbal/Cherry_Medium_dataset.csv"
                },
                "Small": {
                    "model": "models/Ganderbal/Cherry/Small/lstm_Cherry_grade_Small.h5",
                    "dataset": "data/raw/processed/Ganderbal/Cherry_Small_dataset.csv"
                }
            }
        }
    },
    "Narwal": {
        "Apple": {
            "American": {
                "_": {
                    "model": "models/Narwal/American/lstm_American.h5",
                    "dataset": "data/raw/processed/Narwal/American_dataset.csv"
                }
            },
            "Condition": {
                "_": {
                    "model": "models/Narwal/Condition/lstm_Condition.h5",
                    "dataset": "data/raw/processed/Narwal/Condition_dataset.csv"
                }
            },
            "Delicious": {
                "_": {
                    "model": "models/Narwal/Delicious/lstm_Delicious.h5",
                    "dataset": "data/raw/processed/Narwal/Delicious_dataset.csv"
                }
            },
            "Hazratbali": {
                "_": {
                    "model": "models/Narwal/Hazratbali/lstm_Hazratbali.h5",
                    "dataset": "data/raw/processed/Narwal/Hazratbali_dataset.csv"
                }
            },
            "Razakwadi": {
                "_": {
                    "model": "models/Narwal/Razakwadi/lstm_Razakwadi.h5",
                    "dataset": "data/raw/processed/Narwal/Razakwadi_dataset.csv"
                }
            }
        },
        "Cherry": {
            "Cherry": {
                "Large": {
                    "model": "models/Narwal/Cherry/Large/lstm_Cherry_grade_Large.h5",
                    "dataset": "data/raw/processed/Narwal/Cherry_Large_dataset.csv"
                },
                "Medium": {
                    "model": "models/Narwal/Cherry/Medium/lstm_Cherry_grade_Medium.h5",
                    "dataset": "data/raw/processed/Narwal/Cherry_Medium_dataset.csv"
                },
                "Small": {
                    "model": "models/Narwal/Cherry/Small/lstm_Cherry_grade_Small.h5",
                    "dataset": "data/raw/processed/Narwal/Cherry_Small_dataset.csv"
                }
            }
        }
    },
    "Parimpore": {
        "Cherry": {
            "Cherry": {
                "Large": {
                    "model": "models/Parimpore/Cherry/Large/lstm_Cherry_grade_Large.h5",
                    "dataset": "data/raw/processed/Parimpore/Cherry_Large_dataset.csv"
                },
                "Medium": {
                    "model": "models/Parimpore/Cherry/Medium/lstm_Cherry_grade_Medium.h5",
                    "dataset": "data/raw/processed/Parimpore/Cherry_Medium_dataset.csv"
                },
                "Small": {
                    "model": "models/Parimpore/Cherry/Small/lstm_Cherry_grade_Small.h5",
                    "dataset": "data/raw/processed/Parimpore/Cherry_Small_dataset.csv"
                }
            }
        }
    },
    "Pulwama": {
        "Apple": {
            "Pachhar": {
                "American": {
                    "A": {
                        "model": "models/Pulwama/Pachhar/American/A/lstm_American_grade_A.h5",
                        "dataset": "data/raw/processed/Pulwama/Pachhar/American_A_dataset.csv"
                    },
                    "B": {
                        "model": "models/Pulwama/Pachhar/American/B/lstm_American_grade_B.h5",
                        "dataset": "data/raw/processed/Pulwama/Pachhar/American_B_dataset.csv"
                    }
                },
                "Delicious": {
                    "A": {
                        "model": "models/Pulwama/Pachhar/Delicious/A/lstm_Delicious_grade_A.h5",
                        "dataset": "data/raw/processed/Pulwama/Pachhar/Delicious_A_dataset.csv"
                    },
                    "B": {
                        "model": "models/Pulwama/Pachhar/Delicious/B/lstm_Delicious_grade_B.h5",
                        "dataset": "data/raw/processed/Pulwama/Pachhar/Delicious_B_dataset.csv"
                    }
                },
                "Kullu Delicious": {
                    "A": {
                        "model": "models/Pulwama/Pachhar/Kullu Delicious/A/lstm_Kullu Delicious_grade_A.h5",
                        "dataset": "data/raw/processed/Pulwama/Pachhar/Kullu Delicious_A_dataset.csv"
                    },
                    "B": {
                        "model": "models/Pulwama/Pachhar/Kullu Delicious/B/lstm_Kullu Delicious_grade_B.h5",
                        "dataset": "data/raw/processed/Pulwama/Pachhar/Kullu Delicious_B_dataset.csv"
                    }
                }
            },
            "Prichoo": {
                "American": {
                    "A": {
                        "model": "models/Pulwama/Prichoo/American/A/lstm_American_grade_A.h5",
                        "dataset": "data/raw/processed/Pulwama/Prichoo/American_A_dataset.csv"
                    },
                    "B": {
                        "model": "models/Pulwama/Prichoo/American/B/lstm_American_grade_B.h5",
                        "dataset": "data/raw/processed/Pulwama/Prichoo/American_B_dataset.csv"
                    }
                },
                "Delicious": {
                    "A": {
                        "model": "models/Pulwama/Prichoo/Delicious/A/lstm_Delicious_grade_A.h5",
                        "dataset": "data/raw/processed/Pulwama/Prichoo/Delicious_A_dataset.csv"
                    },
                    "B": {
                        "model": "models/Pulwama/Prichoo/Delicious/B/lstm_Delicious_grade_B.h5",
                        "dataset": "data/raw/processed/Pulwama/Prichoo/Delicious_B_dataset.csv"
                    }
                },
                "Kullu Delicious": {
                    "A": {
                        "model": "models/Pulwama/Prichoo/Kullu Delicious/A/lstm_Kullu Delicious_grade_A.h5",
                        "dataset": "data/raw/processed/Pulwama/Prichoo/Kullu Delicious_A_dataset.csv"
                    },
                    "B": {
                        "model": "models/Pulwama/Prichoo/Kullu Delicious/B/lstm_Kullu Delicious_grade_B.h5",
                        "dataset": "data/raw/processed/Pulwama/Prichoo/Kullu Delicious_B_dataset.csv"
                    }
                }
            }
        }
    },
    "Shopian": {
        "Cherry": {
            "Cherry": {
                "Large": {
                    "model": "models/Shopian/Cherry/Large/lstm_Cherry_grade_Large.h5",
                    "dataset": "data/raw/processed/Shopian/Cherry_Large_dataset.csv"
                },
                "Medium": {
                    "model": "models/Shopian/Cherry/Medium/lstm_Cherry_grade_Medium.h5",
                    "dataset": "data/raw/processed/Shopian/Cherry_Medium_dataset.csv"
                },
                "Small": {
                    "model": "models/Shopian/Cherry/Small/lstm_Cherry_grade_Small.h5",
                    "dataset": "data/raw/processed/Shopian/Cherry_Small_dataset.csv"
                }
            }
        },
        "Apple": {
            "American": {
                "A": {
                    "model": "models/Shopian/lstm_American_grade_A.h5",
                    "dataset": "data/raw/processed/Shopian/American_A_dataset.csv"
                },
                "B": {
                    "model": "models/Shopian/lstm_American_grade_B.h5",
                    "dataset": "data/raw/processed/Shopian/American_B_dataset.csv"
                }
            },
            "Delicious": {
                "A": {
                    "model": "models/Shopian/lstm_Delicious_grade_A.h5",
                    "dataset": "data/raw/processed/Shopian/Delicious_A_dataset.csv"
                },
                "B": {
                    "model": "models/Shopian/lstm_Delicious_grade_B.h5",
                    "dataset": "data/raw/processed/Shopian/Delicious_B_dataset.csv"
                }
            },
            "Kullu Delicious": {
                "A": {
                    "model": "models/Shopian/lstm_Kullu Delicious_grade_A.h5",
                    "dataset": "data/raw/processed/Shopian/Kullu Delicious_A_dataset.csv"
                },
                "B": {
                    "model": "models/Shopian/lstm_Kullu Delicious_grade_B.h5",
                    "dataset": "data/raw/processed/Shopian/Kullu Delicious_B_dataset.csv"
                }
            },
            "Maharaji": {
                "A": {
                    "model": "models/Shopian/lstm_Maharaji_grade_A.h5",
                    "dataset": "data/raw/processed/Shopian/Maharaji_A_dataset.csv"
                },
                "B": {
                    "model": "models/Shopian/lstm_Maharaji_grade_B.h5",
                    "dataset": "data/raw/processed/Shopian/Maharaji_B_dataset.csv"
                }
            }
        }
    },
    "Sopore": {
        "Apple": {
            "American": {
                "A": {
                    "model": "models/Sopore/Sopore/American/A/lstm_American_grade_A.h5",
                    "dataset": "data/raw/processed/Sopore/American_A_dataset.csv"
                },
                "B": {
                    "model": "models/Sopore/Sopore/American/B/lstm_American_grade_B.h5",
                    "dataset": "data/raw/processed/Sopore/American_B_dataset.csv"
                }
            },
            "Delicious": {
                "A": {
                    "model": "models/Sopore/Sopore/Delicious/A/lstm_Delicious_grade_A.h5",
                    "dataset": "data/raw/processed/Sopore/Delicious_A_dataset.csv"
                },
                "B": {
                    "model": "models/Sopore/Sopore/Delicious/B/lstm_Delicious_grade_B.h5",
                    "dataset": "data/raw/processed/Sopore/Delicious_B_dataset.csv"
                }
            },
            "Maharaji": {
                "A": {
                    "model": "models/Sopore/Sopore/Maharaji/A/lstm_Maharaji_grade_A.h5",
                    "dataset": "data/raw/processed/Sopore/Maharaji_A_dataset.csv"
                },
                "B": {
                    "model": "models/Sopore/Sopore/Maharaji/B/lstm_Maharaji_grade_B.h5",
                    "dataset": "data/raw/processed/Sopore/Maharaji_B_dataset.csv"
                }
            }
        }
    }
}
