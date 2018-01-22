#include <gtest/gtest.h>

#include <NDT/Tensor.h>
#include <NDT/TupleHelpers.h>

namespace {

using namespace NDT;

TEST(TestInternal, TestTypeListIndex) {

    {
        const bool typeMatch = std::is_same<
                typename internal::TypeListIndex<0,int,float,double>::Type, int>::value;

        ASSERT_TRUE(typeMatch);
    }

    {
        const bool typeMatch = std::is_same<
                typename internal::TypeListIndex<1,int,float,double>::Type, float>::value;

        ASSERT_TRUE(typeMatch);
    }

    {
        const bool typeMatch = std::is_same<
                typename internal::TypeListIndex<2,int,float,double>::Type, double>::value;

        ASSERT_TRUE(typeMatch);
    }

}

TEST(TestInternal, TestTupleSelectiveFloor) {

    std::tuple<float, double, float, int, double> t(0.1f, -1.9, 1.7f, 2, -1.1);

    {
        const bool typeMatch = std::is_same<
                typename internal::TupleTypeSubstitute<0,int,float,double,float,int,double>::Type,
                std::tuple<int,double,float,int,double> >::value;
        ASSERT_TRUE(typeMatch);

        typename internal::TupleTypeSubstitute<0,int,float,double,float,int,double>::Type t2 = t;

        ASSERT_EQ(0, std::get<0>(t2));

    }

    {
        const bool typeMatch = std::is_same<
                typename internal::TupleTypeSubstitute<1,int,float,double,float,int,double>::Type,
                std::tuple<float,int,float,int,double> >::value;
        ASSERT_TRUE(typeMatch);

        typename internal::TupleTypeSubstitute<1,int,float,double,float,int,double>::Type t2 = t;

        ASSERT_EQ(-1, std::get<1>(t2));
    }

    {
        const bool typeMatch = std::is_same<
                typename internal::TupleTypeSubstitute<2,int,float,double,float,int,double>::Type,
                std::tuple<float,double,int,int,double> >::value;
        ASSERT_TRUE(typeMatch);

        typename internal::TupleTypeSubstitute<2,int,float,double,float,int,double>::Type t2 = t;

        ASSERT_EQ(1, std::get<2>(t2));
    }

    {
        const bool typeMatch = std::is_same<
                typename internal::TupleTypeSubstitute<3,int,float,double,float,int,double>::Type,
                std::tuple<float,double,float,int,double> >::value;
        ASSERT_TRUE(typeMatch);

        typename internal::TupleTypeSubstitute<3,int,float,double,float,int,double>::Type t2 = t;

        ASSERT_EQ(2, std::get<3>(t2));
    }

    {
        const bool typeMatch = std::is_same<
                typename internal::TupleTypeSubstitute<4,int,float,double,float,int,double>::Type,
                std::tuple<float,double,float,int,int> >::value;
        ASSERT_TRUE(typeMatch);

        typename internal::TupleTypeSubstitute<4,int,float,double,float,int,double>::Type t2 = t;

        ASSERT_EQ(-1, std::get<4>(t2));
    }

}


} // namespace