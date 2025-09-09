type PromiseReturn<T> = T extends PromiseLike<infer U> ? U : T
type PromiseReturnType<T extends (...args: any[]) => any> = PromiseReturn<
  ReturnType<T>
>

type PartialNested<T> = {
  [P in keyof T]?: T[P] extends object ? Partial<T[P]> : T[P]
}
