import sqlalchemy as sa
import sqlalchemy.orm

Base = sa.orm.declarative_base()

class Obj(Base):
    __tablename__ = 'obj'
    id = sa.Column(sa.Integer, primary_key=True)

class Link(Base):
    def __repr__(self):
        return f'<Link {self.id} / {self.id_src} -> {self.id_dst}>'
    __tablename__ = 'link'
    id = sa.Column(sa.Integer, primary_key=True)
    id_src = sa.Column(sa.Integer, sa.ForeignKey('obj.id'))
    id_dst = sa.Column(sa.Integer, sa.ForeignKey('obj.id'))

class Remap(Base):
    __tablename__ = 'remap'
    id_low = sa.Column(sa.Integer)
    id_high = sa.Column(sa.Integer, autoincrement=False, primary_key=True)

_obj_lowest = sa.select(Remap).where(Remap.id_low == Obj.id).exists()
_obj_query = sa.select(Obj).where(_obj_lowest).subquery()
class FusedObj(Base):
    def __repr__(self):
        return f'<FusedObj {self.id}>'
    __table__ = _obj_query

_F1 = sa.orm.aliased(Remap)
_F2 = sa.orm.aliased(Remap)
_link_query = (
        sa.select(Link.id, _F1.id_low.label('id_src'), _F2.id_low.label('id_dst'))
        .select_from(Link)
        .join(_F1, Link.id_src == _F1.id_high)
        .join(_F2, Link.id_dst == _F2.id_high)
        ).subquery()
class FusedLink(Base):
    def __repr__(self):
        return f'<FusedLink {self.id} / {self.id_src} -> {self.id_dst}>'
    __table__ = _link_query

    src = sa.orm.relationship('FusedObj',
            backref=sa.orm.backref('link_as_src', lazy='raise'),
            primaryjoin='remote(FusedLink.id_src) == foreign(FusedObj.id)',
            viewonly=True)


def main():
    engine = sa.create_engine(f'sqlite://', future=True)
    Base.metadata.create_all(engine)

    sessionmaker = sa.orm.sessionmaker(engine)

    with sessionmaker.begin() as sess:
        for i in range(1, 6):
            sess.execute(sa.insert(Obj).values(id=i))
        for i in range(1, 5):
            sess.execute(sa.insert(Link).values(id_src=i, id_dst=i+1))
        for i in range(2, 4):
            sess.execute(sa.insert(Remap).values(id_low=1, id_high=i))
        for i in range(4, 6):
            sess.execute(sa.insert(Remap).values(id_low=i, id_high=i))

        fused_1 = sess.execute(sa.select(FusedObj).where(FusedObj.id == 1)).scalar()
        # Trouble line!!!
        query = sa.select(FusedLink).where(sa.orm.with_parent(fused_1,
                FusedLink.src))
        print(str(query))
        rel = sess.execute(query).all()
        print('Should only be 2 FusedLink:')
        print(rel)

        print('Underlying objects:')
        print(sess.execute(sa.select(FusedObj)).all())
        print(sess.execute(sa.select(Link)).all())
        print(sess.execute(sa.select(FusedLink)).all())


if __name__ == '__main__':
    main()

