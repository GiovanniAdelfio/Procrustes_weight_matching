import torch

def apply_alignment_MLP(model, proc_dict):
  '''
  The function takes in input the state_dict of a model and the proc_dict produced by the previous function
  it gives as output the same state_dict_model after applying the orthogonals matrices to their corresponding layers
  '''

 model_weights = [['layer0.weight', 'layer0.bias'], ['layer1.weight', 'layer1.bias'], ['layer2.weight', 'layer2.bias'], ['layer3.weight', 'layer3.bias'], ['layer4.weight', 'layer4.bias']]

  for num, layer in enumerate(['layer0.weight', 'layer1.weight', 'layer2.weight', 'layer3.weight']):
      Q = proc_dict[layer].to('cpu')
    # updating the current layer (weights and biases)
      for key in model_weights[num]:
          model[key] = Q.T @ (model[key]).to('cpu')

    # updating the following layer (weights only, no biases)
      if  num < 4 :
          key = model_weights[num+1][0]
          model[key] = model[key].to('cpu') @ Q
        
  return model

def Procrustes(A, B):
  '''
  We use the Procrustes problem to find the optimal orthogonal matrix Q
  which reduces the Frobinius norm 'error' between A and B if applied to A
  '''

  M = A.T @ B
  U, _, V = (torch.linalg.svd(M))     ## I use the SVD decomposition to find U and V transposed
  return (U @ V).to('cuda')           ## I return Q


def proc_weight_matching_MLP(model_a, model_b):
  '''
  We use the function we created earlier to calculate, and save in a dict, the orthogonal matrix we need to align a layer in model_a to the corresponding
  layer in model_b, for every layer of the mlp.
  '''

  proc_dict = {}
  Q = torch.eye(model_a['layer0.weight'].shape[1], device= 'cuda')
  
  for layer in ['layer0.weight', 'layer1.weight', 'layer2.weight', 'layer3.weight']:
    
    model_a_layer = (model_a[layer]).to('cuda') @ Q
    Q = Procrustes(model_a_layer.T, (model_b[layer].T).to('cuda'))
    proc_dict[layer] = Q
    
  return proc_dict

