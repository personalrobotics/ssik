# ikfastpy (deprecation shim)

The `ikfastpy` package was renamed to [`ssik`](https://pypi.org/project/ssik/).

This package is a one-release deprecation shim: importing `ikfastpy` emits a
`DeprecationWarning` and re-exports everything from `ssik`. Update your
imports:

```diff
- import ikfastpy
+ import ssik
```

This shim will be removed in the first post-rename release cycle. Track the
rebuild at <https://github.com/siddhss5/ikfastpy/issues/37>.
