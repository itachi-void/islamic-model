# -*- coding: utf-8 -*-
import unittest
from backend.data.entity_resolver import EntityResolver

class TestEntityResolver(unittest.TestCase):
    def setUp(self):
        self.resolver = EntityResolver()

    def test_exact_resolutions(self):
        # Assert the unit test mappings required by the user
        self.assertEqual(self.resolver.resolve("النبي"), "محمد ﷺ")
        self.assertEqual(self.resolver.resolve("رسول الله"), "محمد ﷺ")
        self.assertEqual(self.resolver.resolve("الصديق"), "أبو بكر")
        self.assertEqual(self.resolver.resolve("الفاروق"), "عمر")
        self.assertEqual(self.resolver.resolve("أبو تراب"), "علي")
        self.assertEqual(self.resolver.resolve("طيبة"), "المدينة")

    def test_prophet_rule(self):
        # Test prophet rule: map 'النبي' -> 'محمد ﷺ' unless another prophet is named
        self.assertEqual(
            self.resolver.expand_query("من هو النبي؟"),
            "من هو النبي؟ محمد ﷺ"
        )
        self.assertEqual(
            self.resolver.expand_query("نبي الله موسى"),
            "نبي الله موسى"
        )

if __name__ == "__main__":
    unittest.main()
