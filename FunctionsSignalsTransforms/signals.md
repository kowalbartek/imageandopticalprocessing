# Signals

Physical processes in nature described mathematically eg. measuring earthquakes (amplitude vs time)

- Discrete representation of any physical process
    - Storing signals in memory is not possible/much more complicated than storing functions.
    - Can't measure continuous signals at discrete points in time.

Discrete instances have an upper bound on how fast we can sample the values.

- The function representation for these will not be in terms of sin/cos/cosin/x^2 etc. but;
    - no concise mathematical description just list of discrete values and where they're positioned
    - `f(x) = [1,2,3,4,5,6,7,8,9,10]` - 10 discrete values (array of integers)
    - This means a function can also just be a list of values.

Signals sampled at discrete intervals = **discrete signals**

-Formal description of Signals:
- `f: N -> R, y = f(x)`
- Single argument of type N (natural number)
- Output of type R (real)

If discrete signals are stored in memory, they are always limited to finite storage no matter the storage size i.e. we
can only represent a finite amount of real numbers. Therefore, we can represent these real numbers as integers, and so
**digital signals** have the form `f: N -> N, y = f(x)`.

We can informally consider them as `f: N -> R, y = f(x)`. This is because we can convert the integers back to real numbers
when we need to.
