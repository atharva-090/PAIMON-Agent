import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/** shadcn's class combiner: clsx for conditionals, tailwind-merge so a later
 *  class wins over an earlier conflicting one instead of both landing. */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
