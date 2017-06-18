gradient = function(y, pred, delta){
  
  diff = pred - y
  g = diff * (delta^2 + diff^2)^(-.5)
  
  return(g)
}

hessian = function(y, pred, delta){
  
  diff = y - pred
  tmp = delta^2 + diff^2
  h = delta^2 * tmp^(-3/2)
  
  return(h)
}

pseudo_huber = function(pred, dtrain, delta = 1){
  
  y = getinfo(dtrain, "label")
  g = gradient(y, pred, delta = delta)
  h = hessian(y, pred, delta = delta)
  
  return(list(grad = g, hess = h))
}