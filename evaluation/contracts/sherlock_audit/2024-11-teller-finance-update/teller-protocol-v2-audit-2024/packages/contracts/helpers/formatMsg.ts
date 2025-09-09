export interface FormatMsgConfig {
  indent?: number
  star?: boolean
  nl?: boolean
  pad?:
    | {
        start: {
          length: number
          char: string
        }
      }
    | {
        end: {
          length: number
          char: string
        }
      }
}
export const formatMsg = (
  msg: string,
  config: FormatMsgConfig = {}
): string => {
  const { indent = 0, star, nl = true } = config

  let result = msg

  if (config.pad) {
    if ('start' in config.pad) {
      const { start } = config.pad
      result = result.padStart(start.length, start.char)
    }
    if ('end' in config.pad) {
      const { end } = config.pad
      result = result.padEnd(end.length, end.char)
    }
  }

  result = result
    .split('\n')
    .map((m) => {
      let r = m
      if (star) r = `* ${m}`
      r = '  '.repeat(indent) + r
      return r
    })
    .join('\n')
  if (nl) result += '\n'

  return result
}
