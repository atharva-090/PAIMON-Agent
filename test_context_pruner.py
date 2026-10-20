"""T1 classify / T2 say-what-changed / T3 keep-the-error, plus cache safety."""
import sys
import types

sys.path.insert(0, r"E:\KIRO projects\Open AI mocker (hermes agent try)\agent")
import context_pruner as P  # noqa: E402

fails = []


class TB:                                   # stand-in TextBlock
    def __init__(self, type="text", text=""):
        self.type, self.text = type, text


class RB:                                   # stand-in tool_result block
    def __init__(self, name, text):
        self.type, self.name = "tool_result", name
        self.output = [TB(text=text)]


def msg(*blocks):
    return types.SimpleNamespace(content=list(blocks))


def page(url, extra="", err=""):
    body = "\n".join(f'- button "Item {i}" [ref=e{i}]' for i in range(1, 40))
    return (f"URL: {url}\n- heading \"Source Attributes\" [ref=e1]\n{body}"
            + (f"\n- StaticText \"{extra}\"" if extra else "")
            + (f"\n- alert \"{err}\" [ref=e99]" if err else ""))


def texts(ctx):
    out = []
    for m in ctx:
        for b in m.content:
            if getattr(b, "type", "") == "tool_result":
                out.append(b.output[0].text)
    return out


A = "https://app.test/newAttribute"
B = "https://app.test/unposted/SOURCE_CATEGORY"

# ---------------------------------------------------------------- T1
print("--- T1: classify by page, not position ---")
ctx = [msg(RB("browse_open", page(A))),
       msg(RB("browse_type", page(A, extra="typed a value"))),
       msg(RB("browse_click_text", page(B))),
       msg(RB("browse_back", page(A, extra="back again")))]
n = P.prune_browser_snapshots(ctx, keep=1)
t = texts(ctx)
print("   collapsed:", n)
for i, x in enumerate(t):
    print("   [%d] %s" % (i, x[:96].replace("\n", " ")))
if "superseded" not in t[0]:
    fails.append("snapshot 0 (page A, read again) not marked superseded")
if "stale" not in t[2]:
    fails.append("snapshot 2 (page B, never returned to) not marked stale")
if t[3].startswith(P.MARKER_PREFIX):
    fails.append("the live snapshot was collapsed")

print("\n--- T1b: A -> B -> back to A keeps the LIVE A expanded ---")
print("   last snapshot still full page:", not t[3].startswith(P.MARKER_PREFIX))
print("   and it is page A          :", P.snapshot_url(t[3]) == A)
if P.snapshot_url(t[3]) != A:
    fails.append("live snapshot is not page A")

# ---------------------------------------------------------------- T2
print("\n--- T2: the marker says what changed ---")
ctx = [msg(RB("browse_open", page(A))),
       msg(RB("browse_type", page(A, extra="Attribute saved successfully"))),
       msg(RB("browse_snapshot", page(A, extra="Attribute saved successfully")))]
P.prune_browser_snapshots(ctx, keep=1)
m0 = texts(ctx)[0]
print("   ", m0[:190].replace("\n", " "))
if "changed after this" not in m0:
    fails.append("T2: no change summary in the marker")
if "Attribute saved successfully" not in m0:
    fails.append("T2: the actual change is not named")

# ---------------------------------------------------------------- T3
print("\n--- T3: the error page survives ---")
ctx = [msg(RB("browse_open", page(A))),
       msg(RB("browse_click_text", page(A, err="Category name already exists"))),
       msg(RB("browse_type", page(A, extra="retyping"))),
       msg(RB("browse_click_text", page(A, extra="retyping again"))),
       msg(RB("browse_snapshot", page(A, extra="current")))]
P.prune_browser_snapshots(ctx, keep=1, error_keep=1)
t = texts(ctx)
for i, x in enumerate(t):
    kind = "FULL " if not x.startswith(P.MARKER_PREFIX) else "mark "
    print("   [%d] %s %s" % (i, kind, x[:86].replace("\n", " ")))
if t[1].startswith(P.MARKER_PREFIX):
    fails.append("T3: the error snapshot was collapsed")
if not t[2].startswith(P.MARKER_PREFIX):
    fails.append("T3: an ordinary old snapshot was not collapsed")

print("\n--- T3b: when an error page DOES age out, the error text survives ---")
ctx = [msg(RB("browse_open", page(A, err="Category name already exists"))),
       msg(RB("browse_click_text", page(A, err="Name is required"))),
       msg(RB("browse_snapshot", page(A, extra="current")))]
P.prune_browser_snapshots(ctx, keep=1, error_keep=1)
t = texts(ctx)
print("   [0]", t[0][:170].replace("\n", " "))
if "already exists" not in t[0]:
    fails.append("T3b: the aged-out error text was lost")

# ------------------------------------------------- cache safety / monotonicity
print("\n--- cache safety: a collapsed snapshot is never rewritten ---")
ctx, seen = [], []
for step in range(1, 9):
    ctx.append(msg(RB("browse_click", page(A, extra="step %d" % step))))
    P.prune_browser_snapshots(ctx, keep=1)
    seen.append(list(texts(ctx)))

stable = True
for step in range(1, len(seen)):
    prev, cur = seen[step - 1], seen[step]
    for i, old in enumerate(prev):
        if old.startswith(P.MARKER_PREFIX) and cur[i] != old:
            stable = False
            fails.append("marker at %d changed on step %d" % (i, step + 1))
print("   8 steps, every frozen marker byte-identical afterwards:", stable)

changes = 0
for step in range(1, len(seen)):
    for i in range(len(seen[step - 1])):
        if seen[step][i] != seen[step - 1][i]:
            changes += 1
print("   total rewrites over 8 steps: %d (one per snapshot, as designed)" % changes)
if changes > 8:
    fails.append("snapshots were rewritten more than once each")

# ---------------------------------------------------------------- savings
print("\n--- what it saves ---")
full = sum(len(x) for x in seen[-1])
raw = len(page(A, extra="step 1")) * len(seen[-1])
print("   8 snapshots kept verbatim : %6d chars" % raw)
print("   after pruning             : %6d chars  (%.0f%% smaller)"
      % (full, 100 * (1 - full / raw)))

# ---------------------------------------------------------------- guards
print("\n--- guards ---")
ctx = [msg(RB("browse_get_text", "the page body text the agent asked for"))]
P.prune_browser_snapshots(ctx, keep=0)
print("   browse_get_text left alone:", not texts(ctx)[0].startswith(P.MARKER_PREFIX))
if texts(ctx)[0].startswith(P.MARKER_PREFIX):
    fails.append("browse_get_text was collapsed")

ctx = [msg(RB("browse_open", page(A)))]
print("   a single snapshot is never collapsed:",
      P.prune_browser_snapshots(ctx, keep=1) == 0)

broken = [msg(types.SimpleNamespace(type="tool_result", name="browse_open",
                                    output=None))]
try:
    P.prune_browser_snapshots(broken, keep=0)
    print("   malformed block survived:", True)
except Exception as e:
    fails.append("raised on a malformed block: %s" % e)

print("\n" + "=" * 68)
if fails:
    print("FAIL:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ALL PASS")
