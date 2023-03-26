interface Model {
  lossFunction: Function;
  parameters: Array<number>;
  learningRate: number;
  maxIterations: number;
  stopCondition: Function;
}

function gradientDescent(model: Model, data: Array<Array<number>>): Array<number> {
  // 1. 定义损失函数
  const lossFunction = model.lossFunction;

  // 2. 初始化参数
  let parameters = model.parameters;

  // 3. 计算梯度
  function computeGradient(data: Array<Array<number>>, parameters: Array<number>): Array<number> {
    let gradient = new Array(parameters.length).fill(0);
    for (let i = 0; i < data.length; i++) {
      let x = data[i].slice(0, -1);
      let y = data[i][data[i].length - 1];
      let prediction = parameters.reduce((acc, cur, idx) => acc + cur * x[idx], 0);
      let error = prediction - y;
      for (let j = 0; j < gradient.length; j++) {
        gradient[j] += error * x[j];
      }
    }
    return gradient;
  }

  // 4. 更新参数
  let learningRate = model.learningRate;
  let maxIterations = model.maxIterations;
  let stopCondition = model.stopCondition;
  let iteration = 0;
  while (iteration < maxIterations && !stopCondition(parameters)) {
    let gradient = computeGradient(data, parameters);
    parameters = parameters.map((p, idx) => p - learningRate * gradient[idx]);
    iteration++;
  }

  // 5. 返回结果
  return parameters;
}
