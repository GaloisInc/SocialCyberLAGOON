"""Long test file for testing message splits.
"""

from lagoon.ingest.toxicity_badwords import _message_split

def test_m1():
    msg = r'''I already chimed in on the issue, but for the list, I'll boil my
comments down to two questions:

1. For anyone who knows: when the documentation refers to "compatibility
with `.time`", is that just saying it was designed that way because
.time returns a float (i.e. for /consistency/ with `.time()`), or is
there some practical reason that you would want `.time()` and
`.mktime()` to return the same type?

2. Mainly for Victor, but anyone can answer: I agree that the natural
output of `mktime()` would be `int` if I were designing it today, but
would there be any /practical/ benefits for making this change? Are
there problems cropping up because it's returning a float? Is it faster
to return an integer?

Best,

Paul

On 4/16/19 10:24 AM, Victor Stinner wrote:
> Hi,
>
> time.mktime() looks "inconsistent" to me and I would like to change
> it, but I'm not sure how it impacts backward compatibility.
> https://bugs.python.org/issue36558
>
> time.mktime() returns a floating point number:
>
>>>> type(time.mktime(time.localtime()))
> <class 'float'>
>
> The documentation says:
>
> "It returns a floating point number, for compatibility with :func:`.time`."
>
> time.time() returns a float because it has sub-second resolution, but
> the C function mktime() returns an integer number of seconds.
>
> Would it make sense to change mktime() return type from float to int?
>
> I would like to change mktime() return type to make the function more
> consistent: all inputs are integers, it sounds wrong to me to return
> float. The result should be integer as well.
>
> How much code would it break? I guess that the main impact are unit
> tests relying on repr(time.mktime(t)) exact value. But it's easy to
> fix the tests: use int(time.mktime(t)) or "%.0f" % time.mktime(t) to
> never get ".0", or use float(time.mktime(t))) to explicitly cast for a
> float (that which be a bad but quick fix).
>
> Note: I wrote and implemented the PEP 564 to avoid any precision loss.
> mktime() will not start loosing precision before year 285,422,891
> (which is quite far in the future ;-)).
>
> Victor'''

    body = _message_split(msg)
    assert body['body'].startswith('I already chimed')
    assert body['sign'].startswith('Best,')
    assert body['quote'].startswith('> Hi,')


def test_m2():
    msg = r'''On Tue, Apr 16, 2019 at 8:19 AM Victor Stinner <vstinner at redhat.com> wrote:

> Le mar. 16 avr. 2019 ? 16:44, Paul Ganssle <paul at ganssle.io> a ?crit :
> > 2. Mainly for Victor, but anyone can answer: I agree that the natural
> output of `mktime()` would be `int` if I were designing it today, but would
> there be any practical benefits for making this change?
>
> It's just for the consistency of the function regarding to C function
> mktime() return type and its input types :-)
>

But all Python times are reported or accept floats -- this allows
sub-second precision without using complicated data structures. None of the
C functions use floats. Consistency with C should not be the issue --
consistency between the time functions is important.


> > Are there problems cropping up because it's returning a float?
>
> None.
>

So let's drop the idea.


> Victor
> --
> Night gathers, and now my watch begins. It shall not end until my death.
> _______________________________________________
> Python-Dev mailing list
> Python-Dev at python.org
> https://mail.python.org/mailman/listinfo/python-dev
> Unsubscribe:
> https://mail.python.org/mailman/options/python-dev/guido%40python.org
>


--
--Guido van Rossum (python.org/~guido)
*Pronouns: he/him/his **(why is my pronoun here?)*
<http://feministing.com/2015/02/03/how-using-they-as-a-singular-pronoun-can-change-the-world/>
-------------- next part --------------
An HTML attachment was scrubbed...
URL: <http://mail.python.org/pipermail/python-dev/attachments/20190416/7c4f2cc1/attachment.html>
'''
    body = _message_split(msg)
    assert body['body'].startswith("But all Python times")
    assert body['body'].endswith("So let's drop the idea.")
    assert body['sign'].startswith("--\n")
    assert body['quote'].startswith("> > 2. Mainly for")
    assert body['quote'].endswith(">")


def test_m3():
    msg = r'''>I would like to change mktime() return type to make the function more
>consistent: all inputs are integers, it sounds wrong to me to return
>float. The result should be integer as well.
In C, the signature of mktime is time_t mktime(struct tm *time);
from Wikipedia, the Unix time_t data type, on many platforms, is a
signed integer, tradionally (32bits). In the newer operating systems,
time_t has been widened to 64 bits.

--
Stéphane Wirtel - https://wirtel.be - @matrixise'''
    body = _message_split(msg)
    assert body['body'].startswith('In C, the signature')
    assert body['quote'].startswith('>I would like')
    assert body['sign'].startswith('--\nSt')


def test_m4():
    msg = r'''On Sat, Nov 19, 2011 at 4:48 PM, Serhiy Storchaka <storchaka at gmail.com> wrote:
> 19.11.11 01:54, Antoine Pitrou ???????(??):
>>
>> Well, the other propositions still seem worse to me. "Qualified" is
>> reasonably accurate, and "qualname" is fairly short and convenient (I
>> would hate to type "__qualifiedname__" or "__qualified_name__" in full).
>> In the same vein, we have __repr__ which may seem weird at first
>> sight :)
>
> What about __reprname__?

Antoine only mentioned 'repr' as being an abbreviation for
'representation', just as 'qualname' will be an abbreviation for
'qualified name'.

The "less ambiguous repr()" use case is just one minor aspect of the
new qualified names, even if it's the most immediately visible, so
using 'repr' in the attribute name would give people all sorts of
wrong ideas about the scope of its utility.

Cheers,
Nick.

--
Nick Coghlan?? |?? ncoghlan at gmail.com?? |?? Brisbane, Australia
'''
    body = _message_split(msg)
    assert body['body'].startswith('Antoine only mentioned')
    assert body['quote'].startswith('> 19.11.11')
    assert body['sign'].startswith('Cheers,\nNick')
    assert 'hate' not in body['body']

