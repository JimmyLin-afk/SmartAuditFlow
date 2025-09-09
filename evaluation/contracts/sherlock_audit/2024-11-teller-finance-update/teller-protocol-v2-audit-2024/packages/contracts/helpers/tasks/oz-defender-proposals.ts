import { subtask } from 'hardhat/config'
import { BatchProposalStep } from 'helpers/hre-extensions'

subtask('oz:defender:get-proposed-steps').setAction(async (args, hre) => {
  const proposedSteps = await hre.deployments
    .readDotFile('.oz-defender-proposed-steps.json')
    .catch((e) => '[]')
    .then<BatchProposalStep[]>((steps) => JSON.parse(steps))
  return proposedSteps
})
subtask('oz:defender:save-proposed-steps').setAction(async (args, hre) => {
  const steps = await hre.run('oz:defender:get-proposed-steps')
  steps.push(...args.steps)
  await hre.deployments.saveDotFile(
    '.oz-defender-proposed-steps.json',
    JSON.stringify(args.steps, null, 2)
  )
})
