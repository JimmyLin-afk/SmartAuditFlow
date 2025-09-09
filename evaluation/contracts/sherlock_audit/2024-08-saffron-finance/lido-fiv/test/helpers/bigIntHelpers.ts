/**
 * Compare two BigInt values within a percentage tolerance.
 *
 * @param {BigInt} value1 - The first value to compare.
 * @param {BigInt} value2 - The second value to compare.
 * @param {number} toleranceBps - The tolerance in basis points.
 * @returns {boolean} - True if the values are within the tolerance, false otherwise.
 */
export function compareWithBpsTolerance(value1: BigInt, value2: BigInt, toleranceBps: number) {
  if (typeof value1 !== 'bigint' || typeof value2 !== 'bigint') {
    throw new Error('Values must be BigInts')
  }

  const ordered = value1 > value2

  // Calculate the difference
  const diff = ordered ? value1 - value2 : value2 - value1

  // Calculate the error margin
  const margin = ((ordered ? value1 : value2) * BigInt(toleranceBps)) / BigInt(10_000)

  // Check if the difference is within the error margin
  return diff <= margin
}

/**
 * Compare two BigInt values within an exact tolerance amount.
 * This is useful if there are very small rounding errors during calculations.
 *
 * @param {BigInt} value1 - The first value to compare.
 * @param {BigInt} value2 - The second value to compare.
 * @param {number} toleranceAmount - The tolerance amount.
 * @returns {boolean} - True if the values are within the tolerance, false otherwise.
 */
export function equalWithTolerance(value1: BigInt, value2: BigInt, toleranceAmount: number) {
  if (typeof value1 !== 'bigint' || typeof value2 !== 'bigint') {
    throw new Error('Values must be BigInts')
  }

  const ordered = value1 > value2

  // Calculate the difference
  const diff = ordered ? value1 - value2 : value2 - value1

  // Check if the difference is within the error margin
  if (diff <= toleranceAmount) {
    return true
  }

  return false
}
