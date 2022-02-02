


def generate_summary(model, tokenizer, source, target):
  
  output = model.generate(source,
                max_length=200, 
                num_beams=10,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True)
  

  output = output
  target[target[:,:]==-100] = 0


  machine_text = [tokenizer.decode(senten,skip_special_tokens=False) for senten in output]
  human_text =  [tokenizer.decode(senten,skip_special_tokens=False) for senten in target ]


  return {"machine_text": machine_text, "human_text":human_text}


