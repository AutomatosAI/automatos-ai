import * as React from "react"

const Separator = React.forwardRef<
  React.ElementRef<"div">,
  React.ComponentPropsWithoutRef<"div">
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={`shrink-0 bg-border h-[1px] w-full ${className || ""}`}
    {...props}
  />
))
Separator.displayName = "Separator"

export { Separator }
