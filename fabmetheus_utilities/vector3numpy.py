"""
High(er) performance vector implementation for fabmetheus,
which should be an easy dropin.

MIT license (c) 2011 Derek Anderson <derek@armyofevilrobots.com>
"""
from __future__ import absolute_import
try:
    import psyco
    psyco.full()
except:
    pass
#Init has to be imported first because it has code to workaround the python bug where relative imports don't work if the module is imported as a main module.
import __init__
import numpy as np

from fabmetheus_utilities import xml_simple_writer
import math
import operator


class V3MetaBase:
    """Base class for metaprogramming magic."""





from fabmetheus_utilities.vector3 import _Vector3

class Vector3:
    """A _FAST_ three dimensional vector class.
    This replaces the stock Vector3 in fabmetheus
    with the numpy one while providing compatibility
    and hopefully the potential reasonable speedup.
    It should be noted; this is SIGNIFICANTLY slower
    than the stock tools. The plan is do refactor dependent
    functions to use the internal _vec instead of the
    individual features, so that we can do
    np.array((_vec, _vec, _vec)) operations later on.
    """


    def __init__(self, x=0.0, y=0.0, z=0.0):
        self._vec = np.array([x, y, z])

    __slots__ = ['x', 'y', 'z']

    @classmethod
    def from_nparray(cls, vec):
        v = cls()
        v._vec = np.array(vec, copy=True)
        return v

    def __abs__(self):
        'Get the magnitude of the Vector3.'
        return np.sqrt(np.vdot(self._vec, self._vec))

    magnitude = __abs__

    def __add__(self, other):
        'Get the sum of this Vector3 and other one.'
        return Vector3( self._vec[0] + other.x, self._vec[1] + other.y, self._vec[2] + other.z )

    def __copy__(self):
        'Get the copy of this Vector3.'
        return Vector3( self._vec[0], self._vec[1], self._vec[2] )

    __pos__ = __copy__

    copy = __copy__

    def __div__(self, other):
        'Get a new Vector3 by dividing each component of this one.'
        return Vector3( self._vec[0] / other, self._vec[1] / other, self._vec[2] / other )


    def __eq__(self, other):
        'Determine whether this vector is identical to other one.'
        if other == None:
            return False
        if other.__class__ != self.__class__:
            return False
        return np.array_equal(self._vec, other._vec)

    def __floordiv__(self, other):
        'Get a new Vector3 by floor dividing each component of this one.'
        return Vector3( self._vec[0] // other, self._vec[1] // other, self._vec[2] // other )

    def _getAccessibleAttribute(self, attributeName):
        'Get the accessible attribute.'
        return attributeNAme in ('x','y','z') and getattr(self, attributeName, None)
        #if attributeName in globalGetAccessibleAttributeSet:
            #return getattr(self, attributeName, None)
        #return None

    def __hash__(self):
        'Determine whether this vector is identical to other one.'
        return self._vec.__hash__()

    def __iadd__(self, other):
        'Add other Vector3 to this one.'
        self._vec[0] += other.x
        self._vec[1] += other.y
        self._vec[2] += other.z
        return self

    def __idiv__(self, other):
        'Divide each component of this Vector3.'
        self._vec[0] /= other
        self._vec[1] /= other
        self._vec[2] /= other
        return self

    def __ifloordiv__(self, other):
        'Floor divide each component of this Vector3.'
        self._vec[0] //= other
        self._vec[1] //= other
        self._vec[2] //= other
        return self

    def __imul__(self, other):
        'Multiply each component of this Vector3.'
        self._vec[0] *= other
        self._vec[1] *= other
        self._vec[2] *= other
        return self

    def __isub__(self, other):
        'Subtract other Vector3 from this one.'
        self._vec[0] -= other.x
        self._vec[1] -= other.y
        self._vec[2] -= other.z
        return self

    def __itruediv__(self, other):
        'True divide each component of this Vector3.'
        self._vec[0] = operator.truediv( self._vec[0], other )
        self._vec[1] = operator.truediv( self._vec[1], other )
        self._vec[2] = operator.truediv( self._vec[2], other )
        return self

    def __mul__(self, other):
        'Get a new Vector3 by multiplying each component of this one.'
        return Vector3( self._vec[0] * other, self._vec[1] * other, self._vec[2] * other )

    def __ne__(self, other):
        'Determine whether this vector is not identical to other one.'
        return not self.__eq__(other)

    def __neg__(self):
        return Vector3( - self._vec[0], - self._vec[1], - self._vec[2] )

    def __nonzero__(self):
        return self._vec[0] != 0 or self._vec[1] != 0 or self._vec[2] != 0

    def __repr__(self):
        'Get the string representation of this Vector3.'
        return '(%s, %s, %s)' % ( self._vec[0], self._vec[1], self._vec[2] )

    def __rdiv__(self, other):
        'Get a new Vector3 by dividing each component of this one.'
        return Vector3( other / self._vec[0], other / self._vec[1], other / self._vec[2] )

    def __rfloordiv__(self, other):
        'Get a new Vector3 by floor dividing each component of this one.'
        return Vector3( other // self._vec[0], other // self._vec[1], other // self._vec[2] )

    def __rmul__(self, other):
        'Get a new Vector3 by multiplying each component of this one.'
        return Vector3( self._vec[0] * other, self._vec[1] * other, self._vec[2] * other )

    def __rtruediv__(self, other):
        'Get a new Vector3 by true dividing each component of this one.'
        return Vector3( operator.truediv( other , self._vec[0] ), operator.truediv( other, self._vec[1] ), operator.truediv( other, self._vec[2] ) )

    def _setAccessibleAttribute(self, attributeName, value):
        'Set the accessible attribute.'
        if attributeName in globalSetAccessibleAttributeSet:
            setattr(self, attributeName, value)

    def __sub__(self, other):
        'Get the difference between the Vector3 and other one.'
        return Vector3( self._vec[0] - other.x, self._vec[1] - other.y, self._vec[2] - other.z )

    def __truediv__(self, other):
        'Get a new Vector3 by true dividing each component of this one.'
        return Vector3( operator.truediv( self._vec[0], other ), operator.truediv( self._vec[1], other ), operator.truediv( self._vec[2], other ) )


    def dot(self, other):
        'Calculate the dot product of this vector with other one.'
        return np.dot(self._vec, other._vec)

    def cross(self, other):
        'Calculate the cross product of this vector with other one.'
        return Vector3.from_nparray(np.cross(self._vec, other._vec))


    def distance(self, other):
        'Get the Euclidean distance between this vector and other one.'
        return np.linalg.norm(self._vec - other._vec)

    def distanceSquared(self, other):
        'Get the square of the Euclidean distance between this vector and other one.'
        return np.square(self.distance(other))

    def dropAxis( self, which = 2 ):
        'Get a complex by removing one axis of the vector3.'
        if which == 0:
            return complex( self._vec[1], self._vec[2] )
        if which == 1:
            return complex( self._vec[0], self._vec[2] )
        if which == 2:
            return complex( self._vec[0], self._vec[1] )

    def getFloatList(self):
        'Get the vector as a list of floats.'
        return [ float( self._vec[0] ), float( self._vec[1] ), float( self._vec[2] ) ]

    def getIsDefault(self):
        'Determine if this is the zero vector.'
        if self._vec[0] != 0.0:
            return False
        if self._vec[1] != 0.0:
            return False
        return self._vec[2] == 0.0

    def getNormalized(self):
        'Get the normalized Vector3.'
        return Vector3.from_nparray(
                self._vec/np.linalg.norm(self._vec))



    def magnitudeSquared(self):
        'Get the square of the magnitude of the Vector3.'
        return self._vec[0] * self._vec[0] + self._vec[1] * self._vec[1] + self._vec[2] * self._vec[2]

    def maximize(self, other):
        'Maximize the Vector3.'
        self._vec[0] =max(other.x, self._vec[0])
        self._vec[1] =max(other.y, self._vec[1])
        self._vec[2] =max(other.z, self._vec[2])

    def minimize(self, other):
        'Minimize the Vector3.'
        self._vec[0] =min(other.x, self._vec[0])
        self._vec[1] =min(other.y, self._vec[1])
        self._vec[2] =min(other.z, self._vec[2])

    def normalize(self):
        self._vec = self._vec/np.linalg.norm(self._vec)

    def reflect( self, normal ):
        'Reflect the Vector3 across the normal, which is assumed to be normalized.'
        distance = 2 * ( self._vec[0] * normal._vec[0] + self._vec[1] * normal._vec[1] + self._vec[2] * normal._vec[2] )
        return Vector3( self._vec[0] - distance * normal._vec[0], self._vec[1] - distance * normal._vec[1], self._vec[2] - distance * normal._vec[2] )

    def setToXYZ(self, x, y, z):
        'Set the x, y, and z components of this Vector3.'
        self._vec = np.array((x, y, z))

    def setToVector3(self, other):
        'Set this Vector3 to be identical to other one.'
        self._vec = np.copy(other._vec)

#Metamagic
for _ofs,_aname in ((0, 'x'), (1, 'y'), (2, 'z')):
    def patchit(ofs, aname):
        """Yes, python scoping rules are effin' weird"""
        def _getter(self, index):
            """herp"""
            return self._vec[index]

        def _setter(self, index, value):
            """derp"""
            self._vec[index] = value

        setattr(Vector3, aname, property(
            lambda s: _getter(s, ofs),
            lambda s,v: _setter(s, ofs, v)))
    patchit(_ofs,_aname)




#UNIT TESTS!
def test_basic():

    v1 = Vector3(1,2,3)
    v2 = Vector3(1,2,3)
    assert v1.x == 1
    assert v1.y == 2
    assert v1.z == 3
    assert v1 == v2





def test_abs():
    """Ensure that we have the same abs outputs."""
    v1 = _Vector3(1,2,99999)
    v2 = Vector3(1,2,99999)
    assert abs(v1) == abs(v2)
    v1 = _Vector3(8,2,9)
    v2 = Vector3(8,2,9)
    assert abs(v1) == abs(v2)
    v1 = _Vector3(8,2,9)
    v2 = Vector3(1,2,3)
    assert abs(v1) != abs(v2)


def test_distance():
    d1 = _Vector3(1,2,3).distance(_Vector3(5,6,7))
    d2 = Vector3(1,2,3).distance(Vector3(5,6,7))
    print d1,d2
    assert d1 == d2



def test_dot():
    v1 = Vector3(1.0, 2.0, 3.0).cross(Vector3(3.0,2.0,1.0))
    v2 = _Vector3(1.0, 2.0, 3.0).cross(_Vector3(3.0,2.0,1.0))
    for i in ['x','y','z']:
        assert(getattr(v1,i) == getattr(v2,i))

def test_normalized():
    v1 = Vector3(3,0,0)
    v2 = Vector3(0,0,0)
    v1.normalize()
    assert v1.x == 1
    assert v1.y == v1.z == 0

def test_add():
    from fabmetheus_utilities.vector3 import _Vector3
    v1 = Vector3(1,2,99999)
    v2 = Vector3(1,2,99999)

    v3 = _Vector3(1,2,99999)
    v4 = _Vector3(1,2,99999)

    v12 = v1 + v2
    v34 = v3 + v4
    print v12
    print v34

    assert (abs(v12) ==
            abs(v34)
            )
