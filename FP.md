## Functions
Properties of Functions:
1. Total
2. Deterministic
3. No side effect


1. Not total: throw an exception or return null
2. Non deterministic: random, clock / time
3. Side effect: readLine, logging


1. Functions        (only use functions, if it's not a function, don't use it)
2. No poly methods  (don't use methods from AnyRef / Java "Object" etc, don't use methods on polymorphic objects)
3. No null          (never use null)
4. No RTTI          (no Runtime Type information/identification, TypeTags etc...)


http://www.lihaoyi.com/post/StrategicScalaStylePracticalTypeSafety.html#scalazzi-scala

Benefits of pure functions
1. immutability
2. concurrency
3. no state


## Typeclasses
A set of 3 things :
1. Types
2. Operations on values of those types
3. Laws governing the operations

In scala : encoded using traits
1. Types are the Type parameters of the trait
2. Operations are the methods of the trait
3. Laws are... comments in the Scaladoc... (at best some Scalacheck testing...)


http://blog.higher-order.com/assets/trampolines.pdf

## Functor, Apply, Applicative, Monad
https://typelevel.org/cats/typeclasses/functor.html#functors-for-effect-management

we can view Functor as the ability to work with a single effect - we can apply a pure function to a single effectful value without needing to “leave” the effect

https://gist.github.com/jdegoes/97459c0045f373f4eaf126998d8f65dc#functor-composition

https://www.sderosiaux.com/articles/2018/08/15/types-never-commit-too-early-part1/

https://www.sderosiaux.com/articles/2018/08/15/types-never-commit-too-early-part2

https://www.sderosiaux.com/articles/2018/08/15/types-never-commit-too-early-part3/?no-cache=1

http://data.tmorris.net/talks/parametricity/4985cb8e6d8d9a24e32d98204526c8e3b9319e33/parametricity.pdf

https://www.youtube.com/watch?v=sxudIMiOo68

https://github.com/pauljamescleary/scala-pet-store

https://leanpub.com/s/CTWecLAq7zJsanbo2l4O5g.pdf (Functional Programming for Mortals with Scalaz - Sam Halliday)

1. Functor - Gives us the power to map values produced by programs without changing their structure
2. Apply - Adds the power to combine two programs into one by combining their values
3. Applicative - Adds the power to produce a 'pure' program that produces the given result
4. Monad - Adds the power to feed the result of one program into a function, which can look at the runtime value and return a new program, which is used to produce the result of the bind

https://github.com/scalaz/scalaz/blob/series/7.3.x/core/src/main/scala/scalaz/Foldable.scala#L9

https://www.youtube.com/watch?v=VCov3HI4jNk

http://julien-truffaut.github.io/Monocle/optics/prism.html

http://degoes.net/articles/fp-vs-oop-part1

## Fiber, Scheduler, Concurrent, Effect, Bracket, Async, Sync, ConcurrentEffect


https://queue.acm.org/detail.cfm?id=2611829 (The Curse of the Excluded Middle - "Mostly functional" programming does not work. - Erik Meijer)

https://jamesclear.com/why-facts-dont-change-minds

https://docs.scala-lang.org/overviews/core/value-classes.html

https://tools.ietf.org/html/rfc2396#appendix-B (Uniform Resource Identifiers (URI): Generic Syntax)

https://github.com/fthomas/refined/blob/5cc30db656142fdc96ab81b0319a76534869d15e/modules/core/shared/src/main/scala/eu/timepit/refined/string.scala#L176

https://github.com/ruippeixotog/scala-scraper

Monoid for error types:
```scala
  final case class ProcessorError[E](error: E, url: URL, html: String)

  final case class Crawl[E, A](error: E, value: A)

  object Crawl {
    implicit def CrawlMonoid[E: Monoid, A: Monoid]: Monoid[Crawl[E, A]] =
      new Monoid[Crawl[E, A]] {
        override def zero: Crawl[E, A] = Crawl(mzero[E], mzero[A])

        override def append(f1: Crawl[E, A], f2: => Crawl[E, A]): Crawl[E, A] =
          Crawl(f1.error |+| f2.error, f1.value |+| f2.value)
      }
  }

  def crawl[E, A: Monoid](seed: Set[URL],
                          processor: (URL, String) => IO[E, A]): IO[Exception, Crawl[List[ProcessorError[E]], A]] = {

    def process1(url: URL, html: String): IO[Nothing, Crawl[List[ProcessorError[E]], A]] =
      processor(url, html).redeemPure(
        e => Crawl(List(ProcessorError(e, url, html)), mzero[A]),
        a => Crawl(Nil, a))

    ???
  }
```

Web Crawler:
```scala
  final case class URL private (url: String) {
    final def relative(page: String): Option[URL] = URL(url + "/" + page)
  }
  object URL {
    def apply(url: String): Option[URL] =
      // TODO: Replace URI.create by something better
      scala.util.Try(java.net.URI.create(url)).toOption match {
        case None => None
        case Some(_) => Some(new URL(url))
      }
  }

  def getURL(url: URL): IO[Exception, String] =
    IO.syncException(scala.io.Source.fromURL(url.url)(scala.io.Codec.UTF8).mkString)

  // TODO: Change to parsing into an immutable data structure
  def extractURLs(root: URL, html: String): List[URL] = {
    val pattern = "href=[\"\']([^\"\']+)[\"\']".r

    scala.util.Try({
      val matches = (for (m <- pattern.findAllMatchIn(html)) yield m.group(1)).toList

      for {
        m   <- matches
        url <- URL(m).toList ++ root.relative(m).toList
      } yield url
    }).getOrElse(Nil)
  }

  final case class ProcessorError[E](error: E, url: URL, html: String)
  final case class Crawl[E, A](error: E, value: A) {
    def leftMap[E2](f: E => E2): Crawl[E2, A] = Crawl(f(error), value)
    def map[A2](f: A => A2): Crawl[E, A2] = Crawl(error, f(value))
  }
  object Crawl {
    implicit def CrawlMonoid[E: Monoid, A: Monoid]: Monoid[Crawl[E, A]] =
      new Monoid[Crawl[E, A]]{
        def zero: Crawl[E, A] = Crawl(mzero[E], mzero[A])
        def append(l: Crawl[E, A], r: => Crawl[E, A]): Crawl[E, A] =
          Crawl(l.error |+| r.error, l.value |+| r.value)
      }
  }

  def crawl[E: Monoid, A: Monoid](
      seeds     : Set[URL],
      router    : URL => Set[URL],
      processor : (URL, String) => IO[E, A]): IO[Exception, Crawl[E, A]] = {
        def loop(seeds: Set[URL], acc: Crawl[E, A]): IO[Exception, Crawl[E, A]] = {
          (IO.traverse(seeds) { url =>
            for {
              html  <- getURL(url)
              crawl <- process1(url, html)
              links = extractURLs(url, html).toSet.flatMap(router)
            } yield (crawl, links)
          }).map(_.foldMap(identity)).flatMap {
            case (crawl0, links) => loop(links, acc |+| crawl0)
          }
        }
  
        def process1(url: URL, html: String): IO[Nothing, Crawl[E, A]] =
          processor(url, html).redeemPure(Crawl(_, mzero[A]), Crawl(mzero[E], _))
  
        loop(seeds, mzero[Crawl[E, A]])
      }

  def crawlE[E, A: Monoid](
    seeds     : Set[URL],
    processor : (URL, String) => IO[E, A]): IO[Exception, Crawl[List[ProcessorError[E]], A]] =
    crawl(seeds, (url, html) => processor(url, html).redeem(
      e => IO.fail(List(ProcessorError(e, url, html))), IO.now))
```

```scala
def crawl[E: Monoid, A: Monoid](
    seed: Set[URL],
    processor: (URL, String) ⇒ IO[E, A],
    criteria: URL ⇒ Boolean,
    maxDepth: Int
  ): IO[Exception, Crawl[E, A]] = {

    def process1(url : URL, html: String): IO[Nothing, Crawl[E, A]] =
      processor(url, html).redeemPure(
        f ⇒ Crawl(f, mzero[A]),
        s ⇒ Crawl(mzero[E], s)
      )

    def processUrl(url : URL, d: Int): IO[Exception, Crawl[E, A]] = for {
      content ← getURL(url)
      res     ← process1(url, content)
      links   = extractURLs(url, content)
      ress    ← if(d <= 0) IO.point(Nil)
                 else IO.parTraverse(links.filter(criteria))(processUrl(_, d - 1))
    } yield (res :: ress).foldMap()


    IO.parTraverse(seed)(processUrl(_, maxDepth)).map(_.foldMap())

  }
```

```scala
implicit class EffectSyntax[F[_, _], E1, A](fea: F[E1, A]) {
    def redeem[E2, B](err: E1 => F[E2, B], succ: A => F[E2, B])(implicit F: Effect[F]): F[E2, B] =
      F.redeem(fea)(err, succ)

    def redeemPure[B](err: E1 => B, succ: A => B)(implicit F: Effect[F]): F[Nothing, B] =
      redeem(
        err.andThen(F.monad[Nothing].point[B](_)),
        succ.andThen(F.monad[Nothing].point[B](_))
      )

  }
```

https://gist.github.com/jdegoes/da80f74c95efe06cb41a929eead532d4 (John De Goes - Crawler Monster)

https://gist.github.com/guizmaii/30368a3d9021c0752bcb58cf4678eaeb

https://gist.github.com/guizmaii/e50c43a5ebe25bbce7db39edfc9a3d61

```scala
def myCode1: Future[Try[Boolean]]

  Task[Either[E, A]]
  F[E, A]

  trait FromFuture[F[_]] {
    def fromFuture[A](fa: => Future[A]): F[A]
  }
  implicit val MyInstance: FromFuture[IO[Throwable, ?]] with MonadError[IO[Throwable, ?], Throwable] {

  }

  def myNewCode[F[_]: MonadError[?, Throwable]: FromFuture]: F[Boolean]

  case class OptionT[F[_], +A](run: F[Option[A]]) {
  case class ErrorT[F[_], +E, +A](run: F[Either[E, A]]) {
    def map[B](f: A => B)(implicit F: Functor[F]): ErrorT[F, E, B] = ???
    def flatMap[B](f: A => ErrorT[F, E, B])(implicit F: Monad[F]): ErrorT[F, E, B] = ???
  }
  object ErrorT {
    def point[F[_]: Applicative, A](a: => A): ErrorT[F, Nothing, A] = 
      ErrorT[F, Nothing, A](Right(a).point[F])
  }
  def myCode1: ErrorT[Future, Error, Unit] = ???
  def myCode2[F[_]: MonadError[Error, ?]]: F[Unit] = ???
  type ErrorfulList[A] = ErrorT[List, Error, A]
  trait MonadError[F[_], E] {
    def fail[A](e: E): F[A]
    def attempt[A](fa: F[A]): F[Either[E, A]]
  }
```

https://www.youtube.com/watch?v=QM86Ab3lL20

https://www.youtube.com/watch?v=knK70T4X7YE

https://github.com/mmenestret/fp-ressources

https://github.com/typelevel/cats-mtl

https://github.com/typelevel/cats-tagless

functional and reactive domain modeling book - https://b-ok2.org/book/3560082/dd33bb/?_ir=1

https://www.youtube.com/watch?v=Eihz7kqn6mU

https://www.slideshare.net/paulszulc/trip-with-monads && https://www.youtube.com/watch?v=hKgnRVQ5Ad8

https://alexn.org/blog/2018/05/06/bifunctor-io.html

http://degoes.net/articles/bifunctor-io

http://degoes.net/articles/effects-without-transformers

https://www.geekabyte.io/2018/05/thoughts-on-dealing-with-having-another.html

https://paperswelove.org/

https://github.com/ProjectSeptemberInc/freek


### Higher order abstract syntax (HOAS) / Recursion schemes / Functors / Strange (Selectable) Functor

https://medium.com/@sinisalouc/overcoming-type-erasure-in-scala-8f2422070d20

https://gist.github.com/jdegoes/f3c0f780cdb3ff293f7a2ee0dfac1ab5 (HOAS)

https://twitter.com/ValentinKasas/status/879414703340081156?s=19

Combine N things =>
1. Same type and N > 1        - use semigroup
2. Same type and N = 0 or 1   - use monoid
3. Obtained independently     - use apply / applicative
4. dependents on each other   - use monad

https://pavkin.ru/reverse-state-monad-in-scala-is-it-possible/

https://github.com/jdegoes/functional-scala

https://www.scala-lang.org/api/2.12.x/index.html

https://github.com/scalaz/scalaz/tree/series/7.3.x/core/src/main/scala/scalaz

https://scalaz.github.io/scalaz-zio/

https://gist.github.com/jdegoes/97459c0045f373f4eaf126998d8f65dc

https://github.com/davegurnell/smartypants

http://www.lihaoyi.com/post/StrategicScalaStylePracticalTypeSafety.html

https://github.com/scalaz/scalazzi

existenital and universal types (_, ? etc)

```scala
object example {
  /**
   * Every type class is a set of three things:
   *
   *   - Types
   *   - Operations on values of those types
   *   - Laws governing the behavior of the operations
   *
   * Every type class instance (or instance, for short) is an implementation
   * of the type class for a set of given types.
   */
  abstract class LessThan[A] {
    // `lessThan` must satisfy transitivity law
    // `lessThan(a, b) && lessThan(b, c) ==> lessThan(a, c)`
    def lessThan(left: A, right: A): Boolean

    final def notLessThan(left: A, right: A): Boolean =
      !lessThan(left, right)
  }
  object LessThan {
    def apply[A](implicit A: LessThan[A]): LessThan[A] = A

    implicit val LessThanInt: LessThan[Int] = new LessThan[Int] {
      def lessThan(left: Int, right: Int): Boolean = left < right
    }
  }
  implicit class LessThanSyntax[A](val l: A) extends AnyVal {
    def < (r: A)(implicit A: LessThan[A]): Boolean =
      A.lessThan(l, r)

    def >= (r: A)(implicit A: LessThan[A]): Boolean =
      A.notLessThan(l, r)
  }
  case class Person(name: String, age: Int)
  object Person {
    implicit val PersonLessThan = new LessThan[Person] {
      def lessThan(left: Person, right: Person): Boolean =
        if (left.name < right.name) true
        else if (left.age < right.age) true
        else false
    }
  }

  def sortAll[A: LessThan](l: List[A]): List[A] = l match {
    case Nil => Nil
    case x :: xs =>
      val (lessThan, notLessThan) = xs.partition(_ < x)

      sortAll(lessThan) ++ List(x) ++ sortAll(notLessThan)
  }

  sortAll(1 :: 3 :: -1 :: 10 :: 45 :: 7 :: 2 :: Nil)
}
```

View bounds and Context bounds ([A <% B], [A : B])

https://www.slideshare.net/jdegoes/scalaz-8-a-whole-new-game

https://adelbertc.github.io/publications/typeclasses-scala17.pdf

https://github.com/aloiscochard/scato

Three properties of functions:
1. Totality. For every single element in the domain, the function returns an element in the codomain.
2. Determinism. If f(a) == b at onme time, then f(a) == b at all times.times
3. No Side Effects. The *only* effect that applying a function has, is computing its return value.

There are three types of type composition:
1. Sum composition - Either, Sealed Trait - sum requires a finite number of terms (vs number of values which is ok to be infinite)
2. Product composition - Tuples, case Classes
3. ADT Algebraic data types (sum and product composition)

Morphisms (functions)
Functions map things from a domain to a co-domain

More on functions
1. Higher order functions: Functions that takes a function as one of its parameters
2. Functional combinator: Functions that only takes functions as its parameters

Mono-morphic functions drawbacks:
1. Not easy to reuse (copy/paste)
2. Many cards to win the game (lots of things can go wrong). This means you have to pay attention, and paying attention is not reliable, we need the compiler to provide the proof.

Polymorphic functions:
Scala doesn't have polymorphic functions and thus we fake them using traits/objects or using functions instead

Type Constructor
1. Tree[A] is not a type, it is a type constructor
2. F(when given one type) => Another Type
3. F(A) => Tree[A]. That is a [*] => * (star to star) kind
4. Either is [*, *] => *
5. * = {x: x is a type in the scala type system}

Partial type application
1. To Turn [*, *] => * to [*] => * We use ? (which requires type projector compiler plugin)
2. type newType[A] = Map[Int, A]
3. trait Foo[A[_, _, _], B, C[_,_]]

Type Classes: Enough but not too much
1. Monomorphic functions know too much
2. Polymorphic functions throw away too much (no more knowledge of comparability for an example)
3. Type classes provide a way to regain structure but as little as possible
4. Passing in a function is an ad-hock alternative

Existential vs. Universal types
```scala
def foo[A] // Universal type

type Foo{
    type S // existential type
}
```

Notes:
In Scala, types and values don't exist in the same universe

https://www.linkedin.com/pulse/know-scala-great-now-learn-functional-programming-john-de-goes

https://gist.github.com/jdegoes/97459c0045f373f4eaf126998d8f65dc

https://gist.github.com/calvinlfer/ed7f70f44725deeb3393759e0583cdcd

https://gist.github.com/MuhammadFarag/1d8e2991f110e3717f413f0db1adf1ba

https://vimeo.com/28793245

https://github.com/scalaz/scalaz/blob/series/7.3.x/core/src/main/scala/scalaz/Foldable.scala

https://gist.github.com/calvinlfer/437c4fcca1e009e317cf44fa5860bb5d

https://mfarag.com/fs-type-classes

https://mfarag.com/

http://julien-truffaut.github.io/Monocle/optics.html

https://gist.github.com/jdegoes/97459c0045f373f4eaf126998d8f65dc#optic-composition-table

https://docs.oracle.com/javase/7/docs/api/java/lang/System.html#exit(int)

https://www.scala-lang.org/api/2.12.3/scala/concurrent/Future$.html

https://www.signifytechnology.com/blog/2018/09/zio-queue-by-wiem-zine-el-abidine

https://gist.github.com/calvinlfer/197d3f4a7aa1365293946da93a24d484

https://gist.github.com/jdegoes/3f51612f8e3e3daa086902f7269721a0

https://gist.github.com/jdegoes/7e7013a8c14e5fcb28f7f00a8245f968

https://gist.github.com/calvinlfer/eeb556cea49bdc7461e645bd6803917c

https://www.iravid.com/posts/fp-and-spark.html

https://softwaremill.com/free-tagless-compared-how-not-to-commit-to-monad-too-early/

https://typelevel.org/blog/2017/12/27/optimizing-final-tagless.html

http://degoes.net/articles/polymorphic-bifunctors

```scala
sealed trait Free[F[_], A] { self =>
      final def map[B](f: A => B): Free[F, B] = self.flatMap(f.andThen(Free.point[F, B](_)))

      final def flatMap[B](f: A => Free[F, B]): Free[F, B] = Free.FlatMap(self, f)

      final def <* [B](that: Free[F, B]): Free[F, A] =
        self.flatMap(a => that.map(_ => a))

      final def *> [B](that: Free[F, B]): Free[F, B] =
        self.flatMap(_ => that)

      final def fold[G[_]: Monad](interpreter: F ~> G): G[A] =
        self match {
          case Free.Return(value0)  => value0().point[G]
          case Free.Effect(fa)      => interpreter(fa)
          case Free.FlatMap(fa0, f) => fa0.fold(interpreter).flatMap(a0 => f(a0).fold(interpreter))
        }
    }
    object Free {
      case class Return[F[_], A](value0: () => A) extends Free[F, A] {
        lazy val value = value0()
      }
      case class Effect[F[_], A](effect: F[A]) extends Free[F, A]
      case class FlatMap[F[_], A0, A](fa0: Free[F, A0], f: A0 => Free[F, A]) extends Free[F, A]

      def point[F[_], A](a: => A): Free[F, A] = Return(() => a)
      def lift[F[_], A](fa: F[A]): Free[F, A] = Effect(fa)
    }

    sealed trait ConsoleF[A]
    final case object ReadLine extends ConsoleF[String]
    final case class PrintLine(line: String) extends ConsoleF[Unit]

    def readLine: Free[ConsoleF, String] = Free.lift[ConsoleF, String](ReadLine)
    def printLine(line: String): Free[ConsoleF, Unit] = Free.lift[ConsoleF, Unit](PrintLine(line))

    val program: Free[ConsoleF, String] =
      for {
        _    <- printLine("Good morning! What is your name?")
        name <- readLine
        _    <- printLine("Good to meet you, " + name + "!")
      } yield name

    import scalaz.zio.IO
    import scalaz.zio.interop.scalaz72._

    val programIO: IO[Nothing, String] =
      program.fold[IO[Nothing, ?]](new NaturalTransformation[ConsoleF, IO[Nothing, ?]] {
        def apply[A](consoleF: ConsoleF[A]): IO[Nothing, A] =
          consoleF match {
            case ReadLine => IO.sync(scala.io.StdIn.readLine())
            case PrintLine(line) => IO.sync(println(line))
          }
      })

    case class TestData(input: List[String], output: List[String])
    case class State[S, A](run: S => (S, A)) {
      def eval(s: S): A = run(s)._2
    }
    object State {
      implicit def MonadState[S]: Monad[State[S, ?]] =
        new Monad[State[S, ?]] {
          def point[A](a: => A): State[S, A] = State(s => (s, a))
          def bind[A, B](fa: State[S, A])(f: A => State[S, B]): State[S, B] =
            State[S, B](s => fa.run(s) match {
              case (s, a) => f(a).run(s)
            })
        }

      def get[S]: State[S, S] = State(s => (s, s))
      def set[S](s: S): State[S, Unit] = State(_ => (s, ()))
      def modify[S](f: S => S): State[S, Unit] =
        get[S].flatMap(s => set(f(s)))
    }

    val programState: State[TestData, String] =
      program.fold[State[TestData, ?]](new NaturalTransformation[ConsoleF, State[TestData, ?]] {
        def apply[A](consoleF: ConsoleF[A]): State[TestData, A] =
          consoleF match {
            case ReadLine =>
              for {
                data <- State.get[TestData]
                line = data.input.head
                _    <- State.set(data.copy(input = data.input.drop(1)))
              } yield line

            case PrintLine(line) =>
              State.modify[TestData](d => d.copy(output = line :: d.output))
          }
      })

    programState.eval(TestData("John" :: Nil, Nil))
```

https://github.com/scalaz/scalaz-reactive

https://stackoverflow.com/questions/7861903/what-are-the-benefits-of-applicative-parsing-over-monadic-parsing

http://doi.org/10.1145/2991041.2991042

https://github.com/facebook/Haxl

https://gist.github.com/ASRagab/433be1f4e9bcd7f851d5e025d71241e6

https://gist.github.com/calvinlfer/321f0cd6a261df021912f991d56c218b#gistcomment-2726359

https://gist.github.com/calvinlfer/eeb556cea49bdc7461e645bd6803917c

https://gist.github.com/calvinlfer/c38e681c3b83a746a8d534482275040c

https://gist.github.com/jdegoes/97459c0045f373f4eaf126998d8f65dc

https://underscore.io/blog/posts/2016/12/05/type-lambdas.html

https://diogocastro.com/blog/2018/10/17/haskells-kind-system-a-primer/

http://learnyouahaskell.com/making-our-own-types-and-typeclasses#kinds-and-some-type-foo

https://www.youtube.com/watch?v=onQSHiafAY8 - ZIO Schedule

https://github.com/Chymyst/curryhoward

https://www.linkedin.com/groups/12166023/

https://leanpub.com/fpmortals

https://typelevel.org/blog/2016/08/21/hkts-moving-forward.html

https://medium.com/bigpanda-engineering/understanding-f-in-scala-4bec5996761f

https://speakerdeck.com/chrisphelps/testing-for-lawful-good-adventurers?slide=9

https://docs.scala-lang.org/tour/upper-type-bounds.html

https://www.youtube.com/watch?v=JZPXzJ5tp9w

https://www.google.com/search?q=scala+newtypes&oq=scala+newtypes

https://www.youtube.com/watch?v=I8LbkfSSR58&list=PLbgaMIhjbmEnaH_LTkxLI7FMa2HsnawM_

https://github.com/data61/fp-course

https://alvinalexander.com/scala/functional-programming-simplified-book

https://www.scala-exercises.org/fp_in_scala/getting_started_with_functional_programming

https://typelevel.org/cats/nomenclature.html

http://blog.ezyang.com/2012/08/applicative-functors/

http://www.adit.io/posts/2013-04-17-functors,_applicatives,_and_monads_in_pictures.html

https://www.linkedin.com/groups/12166023/

https://www.linkedin.com/in/salarrahmanian/

```scala
def crawlIOPar[E: Monoid, A: Monoid](
    seeds     : Set[URL],
    router    : URL => Set[URL],
    processor : (URL, String) => IO[E, A]): IO[Nothing, (Fiber[Nothing, Unit], Ref[(Crawl[E, A], Set[URL])])] =  {

      def start(n: Int, queue: Queue[URL], ref: Ref[(Crawl[E, A], Set[URL])]): IO[Nothing, Fiber[Nothing, Unit]] =
        IO.forkAll(List.fill(n)(queue.take.flatMap(url =>
          getURL(url).redeem(
            _    => IO.unit,
            html => processor(url, html).redeemPure(Crawl(_, mzero[A]), Crawl(mzero[E], _)).flatMap { crawl1 =>
              val urls = extractURLs(url, html).toSet.flatMap(router)

              ref.modify {
                case (crawl0, visited) =>
                  (visited, (crawl0 |+| crawl1, visited ++ urls))
              }.flatMap(old => IO.traverse(urls -- old)(queue.offer(_)).void)
            }
          )
        ).forever)).map(_.map(_ => ()))

      for {
        ref   <- Ref(mzero[Crawl[E, A]] -> seeds)
        queue <- Queue.bounded[URL](1000)
        _     <- IO.sequence(seeds.toList.map(queue.offer))
        fiber <- start(10, queue, ref)
      } yield (fiber, ref)
    }
```

https://www.youtube.com/watch?v=y_QHSDOVJM8

https://github.com/atnos-org/eff

https://www.manning.com/books/type-driven-development-with-idris

https://livebook.manning.com/book/type-driven-development-with-idris/about-this-book/

https://gist.github.com/jdegoes/7d20ed233dfefba70b9da546679e42fc

https://github.com/asweigart/my_first_tic_tac_toe/blob/master/tictactoe.py

http://blog.higher-order.com/assets/fpiscompanion.pdf

https://github.com/slamdata/quasar

FS2, Doobie, ZIO, Scalaz, cats, http4s

https://github.com/scalaz/introduction-to-fp-in-scala

https://github.com/ambiata/introduction-to-fp-in-scala

https://gist.github.com/jdegoes/97459c0045f373f4eaf126998d8f65dc

https://github.com/ovotech/algae#logging

https://arxiv.org/abs/1703.10857

https://github.com/typelevel/cats/blob/master/core/src/main/scala/cats/arrow/Profunctor.scala

https://github.com/scalaz/scalaz/blob/series/7.3.x/core/src/main/scala/scalaz/Profunctor.scala

https://github.com/scalaz/testz

https://github.com/jdegoes/zio-workshop

https://github.com/vivri/Adjective

https://github.com/scalaz/scalaz-zio/blob/master/core/jvm/src/main/scala/scalaz/zio/DefaultRuntime.scala

https://www.reddit.com/r/haskell/comments/36e45c/mtl_is_not_a_monad_transformer_library/

https://www.slideshare.net/jdegoes/orthogonal-functional-architecture

https://www.youtube.com/watch?v=K8OKLorIpZc

https://scalaz.github.io/scalaz/scalaz-2.9.1-6.0.4/doc.sxr/scalaz/Applicative.scala.html

https://typelevel.org/cats/typeclasses/applicative.html

https://underscore.io/books/scala-with-cats/

https://leanpub.com/fpmortals

https://github.com/elbaulp/Scala-Category-Theory

eed3si9n.com/learning-scalaz/

hmemcpy/milewski-ctfp-pdf

https://camo.githubusercontent.com/b4caa27f5df29a8b2c29b1a543ccc4599eb2dae4/68747470733a2f2f63646e2e7261776769742e636f6d2f74706f6c656361742f636174732d696e666f677261706869632f6d61737465722f636174732e7376673f63616368654275737465723d33

http://degoes.net/articles/fp-glossary

https://github.com/softwaremill/sttp

https://github.com/softwaremill/tapir

https://tapir-scala.readthedocs.io/en/latest/index.html

https://github.com/mkotsur/playground-cats-effect/pull/1/files

https://twitter.com/jdegoes/status/1106981861127942144?s=21

https://github.com/pauljamescleary/scala-pet-store/

https://www.innoq.com/en/blog/functional-service-in-scala/

https://tapir-scala.readthedocs.io/en/latest/

https://github.com/mschuwalow/zio-todo-backend

http://degoes.net/articles/testable-zio

https://hackage.haskell.org/package/base-4.12.0.0/docs/Data-Functor.html#v:-36--62-

https://www.youtube.com/watch?v=mkSHhsJXjdc&t=4s

https://github.com/scala-js/scala-js/blob/master/javalanglib/src/main/scala/java/lang/Thread.scala

https://gist.github.com/jdegoes/f3224d96a1f371086b5d240b5c2d5b7a

https://gist.github.com/jdegoes/2f20defd4fe07ff8e2476e92312ba317

https://zio.dev/docs/ecosystem/ecosystem

https://github.com/zio/zio-kafka

https://github.com/scalaz/scalaz-schema/blob/prototyping/modules/core/src/main/scala/SchemaModule.scala 

https://medium.com/@olxc/the-evolution-of-a-scala-programmer-1b7a709fb71f

http://michalostruszka.pl/blog/2015/03/30/scala-case-classes-to-and-from-tuples/

https://gist.github.com/milessabin/fbd9da3361611b91da17

https://www.youtube.com/watch?v=IcgmSRJHu_8

https://github.com/debasishg/frdomain/blob/master/src/main/scala/frdomain/ch6/domain/service/AccountService.scala

http://eed3si9n.com/console-games-in-scala

https://zio.dev/docs/resources/resources

https://github.com/ghostdogpr/zio-cheatsheet

https://t.co/zp1DU0kyn1

https://github.com/r2dbc

https://github.com/zio/zio-kafka/

https://github.com/tabdulradi/zio-instrumentation/

https://medium.com/@wiemzin/zio-with-http4s-and-doobie-952fba51d089

https://gist.github.com/jdegoes/a799c7365d877da1face69dd139466de

https://gist.github.com/jdegoes/a799c7365d877da1face69dd139466de

http://hackage.haskell.org/package/DSTM

## ZIO Fibers

1. Fine-Grained Interruption: You have easy control over the interruptibility status of different regions.
2. Effect Locking: You can lock effects on different thread pools. Unlike Cats IO, ZIO#lock actually respects the thread pool invariant.
3. Lossless Errors. ZIO never loses parallel errors or sequential errors.
4. Execution Traces. ZIO has detailed execution traces showing the steps to a failure and what would have happened without the failure.
5. Fiber Dumps. ZIO can tell you all fibers running in the system, what their status is, and what other fibers they are blocking on.
6. STM. ZIO has software transactional memory that provides composable, interruptible, asynchronous data structures.
7. Increased Polymorphism: All data types in ZIO are polymorphic, which provides strong compile-time guarantees of effect handling.
8. Baked In Typed Error & Reader Effects: These effects have to be added to Cats IO & Monix using EitherT / ErrorT and ReaderT (Kleisli) manually.
9. Guaranteed Finalizers: ZIO guarantees finalizers will run, so long as the effects they are attached to exit (for some reason), while Cats IO and Monix do not make this guarantee.
10. Inherited Monadic Region Status: Forked fibers inherit the settings of parent fibers, while in Cats IO and Monix, they are reset to global defaults.


https://github.com/evolution-gaming/scache

https://github.com/evolution-gaming/scache/blob/master/src/main/scala/com/evolutiongaming/scache/ExpiringCache.scala

http://conal.net/blog/posts/the-c-language-is-purely-functional


https://jrsinclair.com/articles/2019/what-i-wish-someone-had-explained-about-functional-programming/

http://www.tomharding.me/fantasy-land/

https://github.com/fantasyland/fantasy-land