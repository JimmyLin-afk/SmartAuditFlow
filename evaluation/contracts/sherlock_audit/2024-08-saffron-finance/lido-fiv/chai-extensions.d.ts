/// <reference types="chai" />

declare global {
  namespace Chai {
    interface Assertion {
      equalWithTolerance(expected: bigint, overrideTolerance?: number): void
    }
  }
}

declare function equalWithTolerance(expected: bigint, overrideTolerance?: number): Chai.ChaiPlugin
export = equalWithTolerance
