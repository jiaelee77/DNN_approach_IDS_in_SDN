import matplotlib.pyplot as plt
from pandas import DataFrame
import Preprocessor

duration=Preprocessor.training_set1.duration
src_bytes=Preprocessor.training_set1.src_bytes
dst_bytes=Preprocessor.training_set1.dst_bytes
count=Preprocessor.training_set1.count
srv_count=Preprocessor.training_set1.srv_count


norm_duration=Preprocessor.training.duration
plt.plot(norm_duration)

plt.show()