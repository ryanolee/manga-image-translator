import os 
import unicodedata
import sentencepiece as spm
from fairseq.models.transformer import TransformerModel
dir_path = os.path.dirname(os.path.realpath(__file__))

sp_en = spm.SentencePieceProcessor(model_file='spm/spm.en.nopretok.model')
sp_ja = spm.SentencePieceProcessor(model_file='spm/spm.ja.nopretok.model')

model = TransformerModel.from_pretrained(
  '/app/models/checkpoints/',
  checkpoint_file='big.pretrain.pt',
  data_name_or_path='/app/models/data-bin',
  source_lang='ja',
  target_lang='en' 
)

def preprocess_ja(x, alpha=None):
  x = unicodedata.normalize('NFKC', x)
  x = x.strip()
  x = ' '.join(x.split())
  x = sp_ja.encode(x, out_type = int)
  x = ' '.join([sp_ja.IdToPiece(i) for i in x])
  return x

def translate_x(x):
  x = preprocess_ja(x)
  x = model.translate(x)
  print(x)
  x = sp_en.decode(x.split(' '))
  #x = ' '.join(x.split()).replace('▁', '').strip()
  return x

print(translate_x("""
　僕はアイネと共に一度、宿の方に戻った。

　改めて直面するのは部屋の問題――部屋のベッドが一つでは、さすがに狭すぎるだろう。



「じゃあ、僕はまた少し出掛けてくるよ」

「えっ、今戻ったばかりなのに？」

「他の宿で部屋が空いてないか確認するの忘れてたからさ。今ならまだ時間も早いし」



　僕はそう言って部屋を出ようとすると、アイネが僕の服の裾を掴んできた。

　思わず足を止めて、振り返る。
"""))