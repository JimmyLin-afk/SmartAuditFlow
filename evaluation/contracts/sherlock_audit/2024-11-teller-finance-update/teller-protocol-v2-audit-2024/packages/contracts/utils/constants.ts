export const NULL_ADDRESS = `0x${'0'.repeat(40)}`

export enum BidState {
  NONEXISTENT,
  PENDING,
  CANCELLED,
  ACCEPTED,
  PAID,
  LIQUIDATED,
}
